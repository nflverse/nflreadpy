from __future__ import annotations

from typing import Any

__all__: list[str]
__version__: str

def __getattr__(name: str) -> Any: ...

def __dir__() -> list[str]: ...
