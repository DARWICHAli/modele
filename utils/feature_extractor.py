from typing import Dict, Iterable, Callable
from torch import nn, Tensor
import torch
import torchvision.models
class FeatureExtractor(nn.Module):
    """
    Forward hooks are a type of PyTorch hook that can be used to monitor or modify 
    the input and output of a PyTorch model during the forward pass. They are called 
    after the forward pass has been computed, but before the gradient is computed 
    and before the backward pass is run.
    """
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        """Saves the outpoot of a chosen layer

        Args:
            layer_id (str): the layer name of the model.

        Returns:
            fn: the output of the layer.
        """
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features