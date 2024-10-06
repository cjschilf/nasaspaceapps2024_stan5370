`include "macros.sv"


module fir #(
	parameter DATA_SIZE = 64,
	parameter NUM_TAPS = 29,
	parameter [0:NUM_TAPS-1][DATA_SIZE-1:0] COEFFICIENTS = {64'd14069364698352663679, 64'd14079019705160971204, 64'd14081849560299396934, 64'd14078596597701159250, 64'd4851104455826270636, 64'd4855309182591651720, 64'd14075045861369445576, 64'd14083003035537119444, 64'd4855922753288839268, 64'd4869581861842790408, 64'd4873417161314813978, 64'd4872244237957891135, 64'd14080575221256067192, 64'd14097877711263862366, 64'd4884234284023356300, 64'd14097877711263862366, 64'd14080575221256067192, 64'd4872244237957891135, 64'd4873417161314813976, 64'd4869581861842790408, 64'd4855922753288839268, 64'd14083003035537119444, 64'd14075045861369445576, 64'd4855309182591651720, 64'd4851104455826270636, 64'd14078596597701159248, 64'd14081849560299396934, 64'd14079019705160971204, 64'd14069364698352663679},
	parameter DECIMATION = 1
)
(
	input logic clock,
	input logic reset,
	input logic out_full,
	input logic in_empty,
	input logic signed [DATA_SIZE-1:0] in_dout,
	output logic out_wr_en,
	output logic in_rd_en,
	output logic signed [DATA_SIZE-1:0] out_din
);
	logic signed [NUM_TAPS-2:0][DATA_SIZE-1:0] register_vals;
	logic shift_en;
	logic [DATA_SIZE-1:0] shift_in;
	logic [DATA_SIZE-1:0] out_c, out;

	localparam NUM_ADD_LAYERS = 8;
	localparam ADD_BOUND = NUM_TAPS/NUM_ADD_LAYERS;

	typedef enum logic [2:0] {READ, S1, WRITE} state_type;
	state_type state, state_c;

	logic [$clog2(DECIMATION)+1:0] counter, counter_c;
	logic [4:0] stall_counter, stall_counter_c;

	assign shift_in = in_dout;
	
	shiftr #(
		.DATA_SIZE(DATA_SIZE),
		.WIDTH(NUM_TAPS-1)
	)
	sr(
		.clk(clock),
		.shift(shift_en),
		.rst(reset),
		.sr_in(shift_in),
		.sr_out(register_vals)
	);

	always_ff@(posedge clock or posedge reset) begin
		if (reset == 1) begin
			counter <= 0;
			state <= READ;
			stall_counter <= 0;
		end else begin
			counter <= counter_c;
			state <= state_c;
			stall_counter <= stall_counter_c;
		end
	end

	always_comb begin
		counter_c = counter;
		state_c = state;
		out_wr_en = 0;
		in_rd_en = 0;
		shift_en = 0;
		stall_counter_c = stall_counter;
		case (state)
			READ: begin
				if (counter == DECIMATION - 1) begin
					if (in_empty == 1'b0) begin
						counter_c = '0;
						state_c = S1;	
					end
				end else begin
					if (in_empty == 0) begin
						counter_c = counter + 1;
						shift_en = '1;
						in_rd_en = '1;
					end
				end
			end
			S1: begin
				stall_counter_c = stall_counter + 1;
				if (stall_counter == 2) begin
					state_c = WRITE;
					stall_counter_c = '0;
				end
			end
			WRITE: begin
				if (out_full == 0) begin
					out_wr_en = 1'b1;
					in_rd_en = 1'b1;
					shift_en = 1'b1;
					state_c = READ;
				end
			end
		endcase
	end

	logic signed [NUM_TAPS-1:0][DATA_SIZE-1:0] product_layer, product_layer_clk;
	logic signed [NUM_TAPS-1:0][DATA_SIZE-1:0] dequant_product_layer, dequant_product_layer_clk;
	logic signed [NUM_TAPS-1:0][DATA_SIZE-1:0] sum_layer, sum_layer_clk;
	logic signed [DATA_SIZE-1:0] sum_output_clk;

	generate
		if (COEFFICIENTS[0] == 0) begin
			assign product_layer[0] = 0;
			assign dequant_product_layer[0] = 0;
		end else begin
			assign product_layer[0] = $signed(in_dout * $signed(COEFFICIENTS[0]));
			assign dequant_product_layer[0] = `DEQUANTIZE($signed(product_layer_clk[0]));
		end
	endgenerate 
	
	//assign sum_layer[0] = product_layer_clk[0];


	genvar i;
	generate
		for (i = 1; i < NUM_TAPS; i = i + 1) begin : test1
			assign product_layer[i] = $signed(register_vals[i-1]*$signed(COEFFICIENTS[i]));
			assign dequant_product_layer[i] = `DEQUANTIZE($signed(product_layer_clk[i]));
		end
	endgenerate

	logic signed [NUM_ADD_LAYERS-1:0][ADD_BOUND-1:0][DATA_SIZE-1:0] sum_series;
	logic signed [NUM_ADD_LAYERS-1:0][DATA_SIZE-1:0] sum_boundaries;
	logic signed [NUM_ADD_LAYERS-1:0][DATA_SIZE-1:0] sum_boundaries_clk;
	logic signed [NUM_ADD_LAYERS-1:0][DATA_SIZE-1:0] final_add_layer;
	// write into comb, read from clk

	genvar j;
	generate 
		for (i = 0; i < NUM_ADD_LAYERS; i = i + 1) begin : test2
			assign sum_boundaries[i] = sum_series[i][ADD_BOUND-1];
			assign sum_series[i][0] = dequant_product_layer_clk[i*ADD_BOUND];
			
			for (j = 1; j < ADD_BOUND; j = j + 1) begin : test4
				assign sum_series[i][j] = $signed(dequant_product_layer_clk[i*ADD_BOUND + j]) + $signed(sum_series[i][j-1]);
			end
		end
	endgenerate

	assign final_add_layer[0] = sum_boundaries_clk[0];

	generate
		for (i = 1; i < NUM_ADD_LAYERS; i = i + 1) begin : test3
			assign final_add_layer[i] = sum_boundaries_clk[i] + final_add_layer[i-1];
		end
	endgenerate

	always_ff@(posedge reset or posedge clock) begin
		if (reset == 1) begin
			sum_output_clk <= 0;
			product_layer_clk <= 0;
			sum_boundaries_clk <= 0;
			dequant_product_layer_clk <= 0;
		end else begin
			sum_output_clk <= final_add_layer[NUM_ADD_LAYERS-1];
			sum_boundaries_clk <= sum_boundaries;
			product_layer_clk <= product_layer;
			dequant_product_layer_clk <= dequant_product_layer;
		end
	end

	assign out_din = sum_output_clk;
	
endmodule
