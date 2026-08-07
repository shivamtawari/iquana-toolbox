"""Typed instance-segmentation prediction relations.

The existing flat inference API returns ``list[Contour]`` and remains unchanged.
These additive models carry the stable relation identifiers needed by a consumer
that reconstructs hierarchy before assigning database contour IDs.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iquana_toolbox.schemas.database.contours import Contour


class InstanceSegmentationPrediction(BaseModel):
    """One reconstructed contour and its stable prediction-level relation."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(..., min_length=1, description="Stable ID within an inference response.")
    parent_prediction_id: str | None = Field(
        default=None,
        description="Stable prediction ID of the reconstructed parent, if present.",
    )
    label_id: int = Field(..., description="Database label ID for this prediction.")
    score: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    contour: Contour = Field(..., description="Existing toolbox contour geometry payload.")

    @model_validator(mode="after")
    def _validate_relation_fields(self):
        if self.parent_prediction_id == self.prediction_id:
            raise ValueError("A prediction cannot be its own parent.")
        if self.contour.label_id is not None and self.contour.label_id != self.label_id:
            raise ValueError("Prediction label_id must match the contour label_id when it is present.")
        return self


class InstanceSegmentationPredictionResponse(BaseModel):
    """A relation-aware response with referential and cycle validation."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[InstanceSegmentationPrediction] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_prediction_relations(self):
        by_id = {prediction.prediction_id: prediction for prediction in self.predictions}
        if len(by_id) != len(self.predictions):
            raise ValueError("prediction_id values must be unique within a response.")

        for prediction in self.predictions:
            parent_id = prediction.parent_prediction_id
            if parent_id is not None and parent_id not in by_id:
                raise ValueError(
                    f"Prediction {prediction.prediction_id} references missing parent {parent_id}."
                )

        for prediction in self.predictions:
            visited: set[str] = set()
            current: str | None = prediction.prediction_id
            while current is not None:
                if current in visited:
                    raise ValueError("Prediction parent relations must not contain cycles.")
                visited.add(current)
                current = by_id[current].parent_prediction_id
        return self


# Name the relation record explicitly for callers that prefer the contract term.
InstanceSegmentationPredictionRelation = InstanceSegmentationPrediction

