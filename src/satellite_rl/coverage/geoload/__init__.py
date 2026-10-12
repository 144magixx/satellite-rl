"""Pure first-stage geometry and demand-load planning."""
from .config import GeometryConfig, SearchConfig
from .core import Scene, Layout, metrics, validate_layout, initial_layout
