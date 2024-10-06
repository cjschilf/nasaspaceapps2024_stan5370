# Installing dependencies

To run the compiler, the following dependencies are required:

- Torch-MLIR
- CIRCT
- (LLVM/MLIR)

These can be installed by running `./install` in the current directory (`compiler/`).

Installing these requires `cmake`, `git`, and `ninja`. You should probably have Python 3.11 or greater. You also want to have the appropriate `python-dev` package installed.

# Running the compiler
To enable, run `source enable` in the current directory (`compiler/`). This should add all binaries and dependencies to your path.

From here, you can run `pytorch-to-scf` to lower down to SCF, and `scf-to-verilog` to compile to Verilog.
