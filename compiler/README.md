# Installing dependencies

To run the compiler, the following dependencies are required:

- Torch-MLIR
- CIRCT
- (LLVM/MLIR)

These can be installed by running `./install` in the current directory (`compiler/`).

Installing these requires `cmake`, `git`, and `ninja`. If you're messing with PyTorch, you probably have Python installed.

# Running the compiler
To enable, run `source enable` in the current directory (`compiler/`). This should add `torchscript-to-verilog` to your path.

From here, you can run `torchscript-to-verilog <path-to-torchscript-file>` to get a `*.v` file.
