"""Minimal Config class for inference-only mode."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, TypeVar


@dataclass
class Config:
    """Minimal Config for inference-only mode.

    Provides enough functionality to deserialize model configs from JSON
    and build models.
    """

    CLASS_NAME_FIELD = "_CLASS_"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Deserialize from a dictionary."""
        field_info = {f.name: f for f in fields(cls)}
        field_names = set(field_info.keys())
        valid_kwargs = {k: v for k, v in data.items() if k in field_names}
        return cls(**valid_kwargs)

    def as_dict(self) -> dict[str, Any]:
        """Convert to a dictionary."""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                if is_dataclass(value) and not isinstance(value, type):
                    result[field.name] = value.as_dict()
                else:
                    result[field.name] = value
        return result

    def validate(self) -> None:
        """Validate the config. Override in subclasses."""
        pass

    def build(self) -> Any:
        """Build the object this config represents."""
        raise NotImplementedError("Subclasses must implement build()")


C = TypeVar("C", bound=Config)

__all__ = ["Config"]