# nasaspaceapps2024_stan5370

Team stan5370's submission for NASA Space Apps 2024 Chicago.

## Overview

We worked on the "Seismic Detection Across the Solar System" challenge, which required filtering signals from the Mars InSight Lander to identify seismic quakes.

Our solution involves an **FPGA-mounted preprocessor** and a **neural network** on a microcontroller. The preprocessor is designed to maximize efficiency with minimal energy and computing overhead. The neural network identifies 90% of seismic events when they occur.

Independently, it's also possible to forgo the neural network with sufficient data by hand-tuning parameters and adding a second filter. It's also theoretically possible to lower the neural network to mount on an FPGA as well. Either would further reduce system overhead.

## Structure

The neural network (Colin) is in the **home directory** of the repo.

The `verilog` directory (Josh) contains the code for the FPGA preprocessor.

The `TBD` directory (Shannon) contains the experimental code for hand-tuning parameters and additional filters.

The `compiler` directory (Ben) contains experimental code to install and run the pytorch-to-verilog compiler. It doesn't work, but is *very* close to working.
