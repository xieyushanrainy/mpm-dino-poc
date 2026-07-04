"""DINO-conditioned particle-grid dynamics surrogate."""

from .grid import GridSpec, gather_grid, scatter_particles
from .model import ParticleGridSurrogate, SurrogateOutput

__all__ = ["GridSpec", "gather_grid", "scatter_particles", "ParticleGridSurrogate", "SurrogateOutput"]
