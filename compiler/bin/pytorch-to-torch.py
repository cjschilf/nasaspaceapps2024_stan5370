import os
import sys

import torch
from torch_mlir import torchscript

# Make sure venv is activated
cur_dir = os.path.dirname(os.path.abspath(__file__))
activate = cur_dir + "../deps/torch-mlir/venv/bin/activate_this.py"
exec(open(activate).read())

# Load the PyTorch model
path = sys.argv[1]
model = torch.load(path)
model.eval()

# Convert the PyTorch model to torch
example_data = [] # TODO: ADD DATA!!!!!!!!!!!!!
module = torchscript.compile(
        model, example_data, output_type=torchscript.OutputType.TORCH
        )
with open(cur_dir + "temp/torch.mlir", "w+") as f:
    f.write(module.operation.get_asm())
