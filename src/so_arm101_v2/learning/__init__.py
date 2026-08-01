"""Small, explicit learning gates before introducing larger policies."""

from .baselines import BaselineEvaluation, evaluate_baselines
from .decision import build_act_decision_report
from .full_dataset import (
    ImageSignalStatus,
    Lead3TrainingCache,
    SmallModelComparison,
    SmallModelConfig,
    SmallModelKind,
    SmallModelRun,
    build_lead3_training_cache,
    build_small_model,
    classify_image_signal,
    cross_episode_image_indices,
    evaluate_image_ablation,
    load_small_model_comparison,
    train_small_models,
)
from .tiny_model import (
    TinyModelConfig,
    TinyOverfitResult,
    build_memorization_subset,
    run_tiny_overfit,
)
__all__ = [
    "BaselineEvaluation",
    "ImageSignalStatus",
    "Lead3TrainingCache",
    "SmallModelComparison",
    "SmallModelConfig",
    "SmallModelKind",
    "SmallModelRun",
    "TinyModelConfig",
    "TinyOverfitResult",
    "build_memorization_subset",
    "build_lead3_training_cache",
    "build_act_decision_report",
    "build_small_model",
    "classify_image_signal",
    "cross_episode_image_indices",
    "evaluate_baselines",
    "evaluate_image_ablation",
    "load_small_model_comparison",
    "run_tiny_overfit",
    "train_small_models",
]
