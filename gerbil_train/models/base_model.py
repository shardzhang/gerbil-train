"""Abstract base class for all models."""

from abc import ABC, abstractmethod

from torch import nn


class BaseModel(nn.Module, ABC):
    """All models must implement ``validate_fields``."""

    @abstractmethod
    def validate_fields(self, model_cfg) -> None:
        raise NotImplementedError
