"""Public Pydantic contracts shared by toolbox consumers.

The exports are lazy so importing a lightweight contract does not eagerly import
the legacy database/networking schemas (some still carry their own migration
warnings under Pydantic v2).
"""

from importlib import import_module


_PUBLIC_CONTRACTS = {
    "EXCLUSIVE_HIERARCHY_V1": ("iquana_toolbox.schemas.hierarchy", "EXCLUSIVE_HIERARCHY_V1"),
    "HIERARCHY_ARTIFACT_SCHEMA": ("iquana_toolbox.schemas.hierarchy", "HIERARCHY_ARTIFACT_SCHEMA"),
    "HIERARCHY_ARTIFACT_VERSION": ("iquana_toolbox.schemas.hierarchy", "HIERARCHY_ARTIFACT_VERSION"),
    "ExclusiveHierarchyArtifact": ("iquana_toolbox.schemas.hierarchy", "ExclusiveHierarchyArtifact"),
    "HierarchyArtifact": ("iquana_toolbox.schemas.hierarchy", "HierarchyArtifact"),
    "InstanceSegmentationModelInfo": ("iquana_toolbox.schemas.model_info", "InstanceSegmentationModelInfo"),
    "InstanceSegmentationPrediction": ("iquana_toolbox.schemas.inference", "InstanceSegmentationPrediction"),
    "InstanceSegmentationPredictionRelation": (
        "iquana_toolbox.schemas.inference", "InstanceSegmentationPredictionRelation"
    ),
    "InstanceSegmentationPredictionResponse": (
        "iquana_toolbox.schemas.inference", "InstanceSegmentationPredictionResponse"
    ),
    "ModelInfo": ("iquana_toolbox.schemas.model_info", "ModelInfo"),
    "parse_tags_to_model_info": ("iquana_toolbox.schemas.model_info", "parse_tags_to_model_info"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _PUBLIC_CONTRACTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute

__all__ = [
    "EXCLUSIVE_HIERARCHY_V1",
    "HIERARCHY_ARTIFACT_SCHEMA",
    "HIERARCHY_ARTIFACT_VERSION",
    "ExclusiveHierarchyArtifact",
    "HierarchyArtifact",
    "InstanceSegmentationModelInfo",
    "InstanceSegmentationPrediction",
    "InstanceSegmentationPredictionRelation",
    "InstanceSegmentationPredictionResponse",
    "ModelInfo",
    "parse_tags_to_model_info",
]
