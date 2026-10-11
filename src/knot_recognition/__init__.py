__version__ = "0.1.0"

from .dataset import DatasetConfig, KnotImageDataset
from .features import extract_symmetry_invariant_features
from .gauss_pd import GaussPDConfig, GaussPDExtractor, extract_gauss_code
from .model import ResNetConfig, build_resnet, get_resnet
from .preprocess import PreprocessConfig, Preprocessor, preprocess_for_skeleton
from .protein import (
    Crossing,
    ProteinBackbone,
    detect_crossings,
    extract_ca_polyline,
    gauss_code_from_crossings,
    project_polyline,
    sample_viewpoints,
)
from .reidemeister import ReidemeisterConfig, ReidemeisterDetector, detect_moves
from .traverser import AnnotatedImage, KnotTraverser, StrandIntersection, TraversalResult

_LAZY_EXPORTS = {
    "InferenceConfig": ".infer",
    "KnotRecognizer": ".infer",
    "infer_image": ".infer",
    "load_checkpoint": ".infer",
    "TrainConfig": ".train",
    "Trainer": ".train",
    "train": ".train",
    "SolverConfig": ".solver",
    "reduce_skeleton": ".solver",
    "solve_image": ".solver",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_EXPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "AnnotatedImage",
    "KnotTraverser",
    "StrandIntersection",
    "TraversalResult",
    "DatasetConfig",
    "KnotImageDataset",
    "GaussPDConfig",
    "GaussPDExtractor",
    "extract_gauss_code",
    "extract_symmetry_invariant_features",
    "InferenceConfig",
    "KnotRecognizer",
    "infer_image",
    "load_checkpoint",
    "ResNetConfig",
    "build_resnet",
    "get_resnet",
    "PreprocessConfig",
    "Preprocessor",
    "preprocess_for_skeleton",
    "ProteinBackbone",
    "Crossing",
    "extract_ca_polyline",
    "sample_viewpoints",
    "project_polyline",
    "detect_crossings",
    "gauss_code_from_crossings",
    "ReidemeisterConfig",
    "ReidemeisterDetector",
    "detect_moves",
    "TrainConfig",
    "Trainer",
    "train",
    "SolverConfig",
    "reduce_skeleton",
    "solve_image",
]
