"""Versioned metadata for reversible hierarchical instance segmentation."""

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


HIERARCHY_ARTIFACT_SCHEMA = "instance_segmentation_hierarchy"
HIERARCHY_ARTIFACT_VERSION = 1
EXCLUSIVE_HIERARCHY_V1 = "exclusive_hierarchy_v1"


class ExclusiveHierarchyArtifact(BaseModel):
    """The authoritative label topology for an exclusive hierarchy model.

    ``selected_label_ids`` identifies the labels represented by model classes.
    ``label_parent_ids`` is intentionally broader: it is the complete known
    hierarchy (including unselected ancestors) required to validate selected
    paths. Selecting one label is valid; selecting an ancestor and descendant
    together requires every intervening label to be selected as well.

    JSON object keys are strings by definition. Pydantic converts mapping keys
    back to integers during JSON validation, and the post-validation checks then
    validate the reconstructed integer maps and their relationships.
    """

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, serialize_by_alias=True
    )

    schema_version: Literal[HIERARCHY_ARTIFACT_SCHEMA] = Field(
        default=HIERARCHY_ARTIFACT_SCHEMA,
        validation_alias=AliasChoices("schema_version", "schema"),
        serialization_alias="schema",
        description="Stable artifact schema identifier. Serialized as the legacy 'schema' key.",
    )
    version: Literal[HIERARCHY_ARTIFACT_VERSION] = Field(
        default=HIERARCHY_ARTIFACT_VERSION,
        description="Version of the artifact schema.",
    )
    target_encoding: Literal[EXCLUSIVE_HIERARCHY_V1] = Field(
        default=EXCLUSIVE_HIERARCHY_V1,
        description="Encoding used for mutually exclusive hierarchical targets.",
    )
    selected_label_ids: list[int] = Field(
        ..., description="Database label IDs exported into this model."
    )
    database_label_id_to_model_class_index: dict[int, int] = Field(
        ..., description="Database label ID to contiguous model class index."
    )
    model_class_index_to_database_label_id: dict[int, int] = Field(
        ..., description="Inverse model class index to database label ID."
    )
    label_parent_ids: dict[int, int | None] = Field(
        ...,
        validation_alias=AliasChoices(
            "label_parent_ids", "database_label_id_to_parent_label_id"
        ),
        description=(
            "Complete known label hierarchy: every key is a known database label ID "
            "and its value is its immediate parent, or null for a root. This map "
            "includes unselected ancestors needed to validate selected paths; class "
            "maps remain selected-label-only."
        ),
    )

    @property
    def database_to_model(self) -> dict[int, int]:
        """Compatibility spelling used by the current AI label mapping."""
        return self.database_label_id_to_model_class_index

    @property
    def model_to_database(self) -> dict[int, int]:
        """Compatibility spelling used by the current AI label mapping."""
        return self.model_class_index_to_database_label_id

    @property
    def label_parent_map(self) -> dict[int, int | None]:
        """Short access to the authoritative immediate-parent map."""
        return self.label_parent_ids

    @property
    def database_label_id_to_parent_label_id(self) -> dict[int, int | None]:
        """Backward-compatible spelling for the complete label hierarchy map."""
        return self.label_parent_ids

    @model_validator(mode="after")
    def _validate_topology(self):
        selected = self.selected_label_ids
        database_to_model = self.database_label_id_to_model_class_index
        model_to_database = self.model_class_index_to_database_label_id
        parent_map = self.label_parent_ids

        if len(set(selected)) != len(selected):
            raise ValueError("selected_label_ids must contain unique database label IDs.")

        selected_set = set(selected)
        database_ids = _validate_integer_map_keys(database_to_model, "database label")
        _validate_integer_map_keys(model_to_database, "model class")
        _validate_integer_values(database_to_model, "database-to-model class")
        _validate_integer_values(model_to_database, "model-to-database label")
        class_indices = list(database_to_model.values())

        if database_ids != selected_set:
            raise ValueError(
                "selected_label_ids must exactly match the database-to-model map keys."
            )
        expected_model_to_database = {
            class_index: database_id
            for database_id, class_index in database_to_model.items()
        }
        if model_to_database != expected_model_to_database:
            raise ValueError(
                "database_label_id_to_model_class_index and "
                "model_class_index_to_database_label_id must be exact inverses "
                "with no extra keys."
            )
        expected_class_indices = set(range(len(selected)))
        if set(class_indices) != expected_class_indices:
            raise ValueError(
                "Model class indices must be exactly contiguous from 0 through N-1."
            )

        parent_ids = _validate_integer_map_keys(parent_map, "parent-map label")
        if not parent_ids:
            raise ValueError("label_parent_ids must contain the complete known hierarchy.")
        for database_id, parent_id in parent_map.items():
            if parent_id is not None and (
                not isinstance(parent_id, int) or isinstance(parent_id, bool)
            ):
                raise ValueError("Parent label IDs must be integers or null.")
            if parent_id is not None and parent_id not in parent_ids:
                raise ValueError(
                    f"Label {database_id} references missing parent label {parent_id}."
                )

        _validate_parent_cycles(parent_map)

        unknown_selected = selected_set - parent_ids
        if unknown_selected:
            raise ValueError(
                "selected_label_ids must reference labels in the complete hierarchy: "
                f"missing {sorted(unknown_selected)}."
            )
        _validate_selected_paths(selected_set, parent_map)
        return self


# A concise public alias for callers that do not need to repeat the encoding name.
HierarchyArtifact = ExclusiveHierarchyArtifact


def _validate_integer_map_keys(mapping: dict[int, object], description: str) -> set[int]:
    keys = set(mapping)
    if any(not isinstance(key, int) or isinstance(key, bool) for key in keys):
        raise ValueError(f"{description} map keys must be integers after JSON decoding.")
    return keys


def _validate_integer_values(mapping: dict[int, object], description: str) -> None:
    if any(not isinstance(value, int) or isinstance(value, bool) for value in mapping.values()):
        raise ValueError(f"{description} map values must be integers.")


def _validate_parent_cycles(parent_map: dict[int, int | None]) -> None:
    """Reject cycles while following every known label's parent chain."""
    for start in parent_map:
        visited: set[int] = set()
        current: int | None = start
        while current is not None:
            if current in visited:
                raise ValueError("The selected label hierarchy must not contain cycles.")
            visited.add(current)
            current = parent_map[current]


def _validate_selected_paths(
    selected_label_ids: set[int], parent_map: dict[int, int | None]
) -> None:
    """Require selected labels to include every level between selected ancestors."""
    for descendant_id in selected_label_ids:
        current = parent_map[descendant_id]
        intervening: list[int] = []
        while current is not None:
            if current in selected_label_ids:
                if intervening:
                    raise ValueError(
                        "selected label scope skips intermediate label(s) "
                        f"{list(reversed(intervening))} between selected ancestor "
                        f"{current} and descendant {descendant_id}."
                    )
                break
            intervening.append(current)
            current = parent_map[current]
