from typing import Mapping, Optional, Iterator, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


class Trainer(nn.Module):
    """ Trainer for classification models using PyTorch."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr)
        self.loss_fn = loss_fn
        self.device = device

    def train(self, dataloader: DataLoader, *, epochs: int = 100, silent: bool = False) -> None:
        for _ in self.train_iter(dataloader, epochs=epochs, silent=silent):
            pass

    def train_iter(
        self,
        dataloader: DataLoader,
        *,
        epochs: int = 100,
        silent: bool = False,
    ) -> Iterator[nn.Module]:

        model = self.model.to(self.device)
        self._optimizer_to(self.optimizer, self.device)

        for epoch in tqdm(range(epochs)):
            model.train()
            for data, target in dataloader:
                x = data.to(self.device)
                y = target.to(self.device)

                self.optimizer.zero_grad()
                outputs = model(x)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()

            yield model  
            
    def _optimizer_to(self, optim_: torch.optim.Optimizer, device: torch.device) -> None:
        """Moves optimizer state tensors to the device."""
        for param in optim_.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["optimizer"] = self.optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        del state_dict["optimizer"]
        super().load_state_dict(state_dict, strict, assign)