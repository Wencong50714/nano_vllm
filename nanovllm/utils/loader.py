import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from collections.abc import Generator


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def get_all_weights(path: str) -> Generator[tuple[str, torch.Tensor], None, None]:
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                tensor = f.get_tensor(weight_name)
                yield (weight_name, tensor)

def load_model(model: nn.Module, path: str):
    model.load_weights(get_all_weights(path), default_weight_loader)
