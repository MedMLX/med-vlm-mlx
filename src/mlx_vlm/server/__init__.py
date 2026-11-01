"""Medical text-generation HTTP service."""

from mlx_vlm.server.app import app
from mlx_vlm.server.cli import main

__all__ = ["app", "main"]
