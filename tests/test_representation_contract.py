from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch


import pytest
from pydantic import ValidationError

from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas import (
    ExclusiveHierarchyArtifact as ExportedHierarchyArtifact,
    InstanceSegmentationPredictionResponse as ExportedPredictionResponse,
)
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.hierarchy import ExclusiveHierarchyArtifact
from iquana_toolbox.schemas.inference import (
    InstanceSegmentationPrediction,
    InstanceSegmentationPredictionResponse,
)
from iquana_toolbox.schemas.model_info import (
    InstanceSegmentationModelInfo,
    parse_tags_to_model_info,
)


def _model_payload(**overrides):
    payload = {
        "registry_key": "mask2former",
        "name": "Mask2Former",
        "description": "instance model",
        "usage_tip": "use it",
        "task": "instance_segmentation",
        "label_ids": [7, 42],
    }
    payload.update(overrides)
    return payload


def _artifact_payload(**overrides):
    payload = {
        "selected_label_ids": [10, 20, 30],
        "database_label_id_to_model_class_index": {10: 0, 20: 1, 30: 2},
        "model_class_index_to_database_label_id": {0: 10, 1: 20, 2: 30},
        "database_label_id_to_parent_label_id": {10: None, 20: 10, 30: 20},
    }
    payload.update(overrides)
    return payload


def _artifact(**overrides):
    return ExclusiveHierarchyArtifact.model_validate(_artifact_payload(**overrides))


def _contour(label_id: int) -> Contour:
    return Contour(label_id=label_id, x=[0.1, 0.2, 0.2], y=[0.1, 0.1, 0.2])


def test_old_model_payload_defaults_to_flat_standard():
    info = InstanceSegmentationModelInfo.model_validate(_model_payload())
    assert info.segmentation_mode == "flat"
    assert info.target_encoding == "standard"
    assert info.tags["segmentation_mode"] == "flat"
    assert info.tags["target_encoding"] == "standard"

    old_tags = parse_tags_to_model_info(_model_payload(task="instance-segmentation"))
    assert old_tags.segmentation_mode == "flat"
    assert old_tags.target_encoding == "standard"


def test_representation_tag_mirror_updates_after_valid_assignment():
    info = InstanceSegmentationModelInfo.model_validate(_model_payload())

    info.segmentation_mode = "hierarchical"
    info.target_encoding = "exclusive_hierarchy_v1"

    assert info.tags["segmentation_mode"] == "hierarchical"
    assert info.tags["target_encoding"] == "exclusive_hierarchy_v1"


def test_registry_registration_and_read_path_use_current_representation_tags():
    info = InstanceSegmentationModelInfo.model_validate(_model_payload())
    info.info_url = "https://example.test/model"
    info.badges = ["accurate", "hierarchical"]
    info.trainable = True
    info.training_parameters = [{
        "key": "epochs",
        "label": "Epochs",
        "default_value": 10,
        "type": "int",
    }]
    info.architecture = "Mask2Former"
    info.license = "Apache-2.0"
    info.input_resolution = [1024, 1024]
    info.performance = {"num_parameters": 1200000, "latency_ms": 12.5}
    info.tags = {"team": "vision", "name": "user-value-must-not-win"}
    model = SimpleNamespace(model_info=info, get_artifacts=lambda _tmp_dir: None)

    class FakeMlflowClient:
        def __init__(self):
            self.models = {}

        def search_registered_models(self, filter_string=None):
            models = list(self.models.values())
            if filter_string:
                return [
                    model for model in models
                    if all(
                        model.tags.get(part.split(" = ", 1)[0].removeprefix("tags."))
                        == part.split(" = ", 1)[1].strip("'")
                        for part in filter_string.split(" AND ")
                    )
                ]
            return models

        def get_registered_model(self, name):
            return self.models[name]

        def set_registered_model_tag(self, name, key, value):
            model = self.models.setdefault(name, SimpleNamespace(name=name, tags={}))
            model.tags[key] = value

        def update_registered_model(self, name, description=None):
            self.models[name].description = description

        def set_registered_model_alias(self, name, alias, version):
            pass


    fake_client = FakeMlflowClient()
    run_context = MagicMock()
    run_context.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1")
    )
    run_context.__exit__.return_value = False

    with patch("iquana_toolbox.mlflow.MlflowClient", return_value=fake_client), \
            patch("iquana_toolbox.mlflow.mlflow.set_tracking_uri"), \
            patch("iquana_toolbox.mlflow.mlflow.start_run", return_value=run_context), \
            patch("iquana_toolbox.mlflow.mlflow.pyfunc.log_model") as log_model:
        registry = MLFlowModelRegistry("http://example")
        registry.register_model(model)

        registered_tags = registry.client.get_registered_model("mask2former").tags
        assert log_model.call_args.kwargs["tags"] == registered_tags
        assert registered_tags["segmentation_mode"] == "flat"
        assert registered_tags["target_encoding"] == "standard"
        assert registered_tags["label_ids"] == "[7,42]"
        assert registered_tags["training_parameters"] == (
            '[{"default_value":10,"description":"","key":"epochs","label":"Epochs",'
            '"max_value":null,"min_value":null,"options":null,"step":null,"type":"int"}]'
        )
        assert registered_tags["name"] == "Mask2Former"
        assert registered_tags["team"] == "vision"

        restored = registry.get_model_info("mask2former")
        assert isinstance(restored, InstanceSegmentationModelInfo)
        assert restored.registry_key == "mask2former"
        assert restored.name == "Mask2Former"
        assert restored.description == "instance model"
        assert restored.usage_tip == "use it"
        assert restored.info_url == "https://example.test/model"
        assert restored.badges == ["accurate", "hierarchical"]
        assert restored.trainable is True
        assert restored.architecture == "Mask2Former"
        assert restored.license == "Apache-2.0"
        assert restored.input_resolution == [1024, 1024]
        assert restored.performance.latency_ms == 12.5
        assert restored.tags["team"] == "vision"
        assert restored.tags["name"] == "user-value-must-not-win"
        assert restored.segmentation_mode == "flat"
        assert restored.target_encoding == "standard"

        # The already-registered path must publish valid post-construction mutations.
        info.segmentation_mode = "hierarchical"
        info.target_encoding = "exclusive_hierarchy_v1"
        registry.register_model(model)
        restored = registry.get_model_info("mask2former")
        expected = {
            "segmentation_mode": "hierarchical",
            "target_encoding": "exclusive_hierarchy_v1",
        }
        registered_tags = registry.client.get_registered_model("mask2former").tags
        assert {key: registered_tags[key] for key in expected} == expected
        assert restored.segmentation_mode == "hierarchical"
        assert restored.target_encoding == "exclusive_hierarchy_v1"



@pytest.mark.parametrize(
    ("overrides", "expected_task"),
    [
        ({"task": "not-a-task"}, None),
        ({"model_task": "not-a-task"}, None),
        ({"task": "instance_segmentation", "model_task": "prompted_segmentation"}, None),
        ({"task": "instance-segmentation", "model_task": "instance"}, "instance_segmentation"),
        ({"task": None}, "instance_segmentation"),
    ],
)
def test_task_discriminators_are_strict_and_backward_compatible(overrides, expected_task):
    payload = _model_payload()
    payload.pop("task", None)
    payload.update(overrides)

    if expected_task is None:
        with pytest.raises(ValueError, match="Unknown|Conflicting"):
            parse_tags_to_model_info(payload)
    else:
        parsed = parse_tags_to_model_info(payload)
        assert isinstance(parsed, InstanceSegmentationModelInfo)


def test_representation_metadata_conflicts_with_non_instance_task():
    with pytest.raises(ValueError, match="only be declared"):
        parse_tags_to_model_info(
            _model_payload(task="prompted_segmentation", segmentation_mode="flat")
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("segmentation_mode", "unknown"), ("target_encoding", "exclusive_hierarchy_v2")],
)
def test_invalid_representation_literals_are_rejected(field, value):
    with pytest.raises(ValidationError):
        InstanceSegmentationModelInfo.model_validate(_model_payload(**{field: value}))


def test_mlflow_tags_parse_and_round_trip_representation_fields():
    parsed = parse_tags_to_model_info(
        _model_payload(
            segmentation_mode="hierarchical",
            target_encoding="exclusive_hierarchy_v1",
        )
    )
    assert isinstance(parsed, InstanceSegmentationModelInfo)
    assert parsed.segmentation_mode == "hierarchical"
    assert parsed.target_encoding == "exclusive_hierarchy_v1"
    assert parsed.tags["segmentation_mode"] == "hierarchical"
    assert parsed.tags["target_encoding"] == "exclusive_hierarchy_v1"

    round_tripped = parse_tags_to_model_info(
        parsed.model_dump() | {"task": "instance_segmentation"}
    )
    assert round_tripped.segmentation_mode == "hierarchical"
    assert round_tripped.target_encoding == "exclusive_hierarchy_v1"


def test_invalid_declared_instance_segmentation_tags_do_not_fall_back():
    with pytest.raises(ValidationError):
        parse_tags_to_model_info(_model_payload(target_encoding="not-a-version"))

    with pytest.raises(ValidationError):
        parse_tags_to_model_info(
            _model_payload(
                task=None,
                prompt_types_supported=["box"],
                refinement_supported=False,
                segmentation_mode="not-a-mode",
            )
        )


def test_hierarchy_artifact_json_round_trip_restores_integer_ids():
    artifact = _artifact()
    restored = ExclusiveHierarchyArtifact.model_validate_json(artifact.model_dump_json())
    assert restored.database_label_id_to_model_class_index == {10: 0, 20: 1, 30: 2}
    assert all(isinstance(key, int) for key in restored.database_label_id_to_parent_label_id)
    assert restored.model_class_index_to_database_label_id == {0: 10, 1: 20, 2: 30}


def test_hierarchy_artifact_rejects_inverse_map_mismatch():
    with pytest.raises(ValidationError, match="exact inverses"):
        _artifact(model_class_index_to_database_label_id={0: 10, 1: 30, 2: 20})


def test_hierarchy_artifact_rejects_duplicate_or_negative_class_indices():
    with pytest.raises(ValidationError, match="exact inverses"):
        _artifact(
            database_label_id_to_model_class_index={10: 0, 20: 0, 30: 1},
            model_class_index_to_database_label_id={0: 10, 1: 30, 2: 20},
        )
    with pytest.raises(ValidationError, match="contiguous"):
        _artifact(
            database_label_id_to_model_class_index={10: 0, 20: 1, 30: 3},
            model_class_index_to_database_label_id={0: 10, 1: 20, 3: 30},
        )


def test_hierarchy_artifact_rejects_extra_inverse_map_keys():
    with pytest.raises(ValidationError, match="no extra keys"):
        _artifact(
            model_class_index_to_database_label_id={0: 10, 1: 20, 2: 30, 99: 10}
        )


def test_hierarchy_artifact_rejects_unknown_version():
    with pytest.raises(ValidationError):
        _artifact(version=2)


def test_hierarchy_artifact_rejects_missing_parent_and_skipped_level():
    with pytest.raises(ValidationError, match="missing parent label"):
        _artifact(database_label_id_to_parent_label_id={10: None, 20: 10, 30: 99})

    with pytest.raises(ValidationError, match="skips intermediate"):
        _artifact(
            selected_label_ids=[10, 30],
            database_label_id_to_model_class_index={10: 0, 30: 1},
            model_class_index_to_database_label_id={0: 10, 1: 30},
            database_label_id_to_parent_label_id={10: None, 20: 10, 30: 20},
        )


def test_hierarchy_artifact_rejects_cycles():
    with pytest.raises(ValidationError, match="cycles"):
        _artifact(database_label_id_to_parent_label_id={10: 20, 20: 10, 30: 20})


def test_single_label_hierarchy_is_valid():
    artifact = _artifact(
        selected_label_ids=[30],
        database_label_id_to_model_class_index={30: 0},
        model_class_index_to_database_label_id={0: 30},
        database_label_id_to_parent_label_id={10: None, 20: 10, 30: 20},
    )
    assert artifact.selected_label_ids == [30]
    assert artifact.label_parent_ids == {10: None, 20: 10, 30: 20}


def test_hierarchy_schema_version_and_public_exports():
    artifact = _artifact()
    assert artifact.schema_version == "instance_segmentation_hierarchy"
    assert artifact.model_dump()["schema"] == "instance_segmentation_hierarchy"
    assert artifact.model_dump(by_alias=True)["schema"] == "instance_segmentation_hierarchy"
    assert ExclusiveHierarchyArtifact is ExportedHierarchyArtifact
    assert InstanceSegmentationPredictionResponse is ExportedPredictionResponse


def test_inference_relation_validates_parent_references_and_geometry():
    parent = InstanceSegmentationPrediction(
        prediction_id="p1", label_id=10, score=0.9, contour=_contour(10)
    )
    child = InstanceSegmentationPrediction(
        prediction_id="c1", parent_prediction_id="p1", label_id=20, score=0.8,
        contour=_contour(20),
    )
    response = InstanceSegmentationPredictionResponse(predictions=[child, parent])
    assert response.predictions[0].parent_prediction_id == "p1"

    with pytest.raises(ValidationError, match="missing parent"):
        InstanceSegmentationPredictionResponse(
            predictions=[child.model_copy(update={"parent_prediction_id": "missing"})]
        )

    with pytest.raises(ValidationError, match="label_id"):
        InstanceSegmentationPrediction(
            prediction_id="bad", label_id=20, score=0.8, contour=_contour(10)
        )

    with pytest.raises(ValidationError, match="cycles"):
        cycle_parent = parent.model_copy(update={"parent_prediction_id": "c1"})
        InstanceSegmentationPredictionResponse(predictions=[cycle_parent, child])


def test_generate_trained_model_registry_key_format():
    from iquana_toolbox.schemas.model_info import generate_trained_model_registry_key
    key = generate_trained_model_registry_key(42, "2db4b6a0-3c75-4f74-97cf-15eed5d1c338")
    assert key == "mask2former-ds42-2db4b6a0-3c75-4f74-97cf-15eed5d1c338"


def test_model_role_default_and_overrides():
    info_default = InstanceSegmentationModelInfo.model_validate(_model_payload())
    assert info_default.model_role == "training_base"

    info_trained = InstanceSegmentationModelInfo.model_validate(_model_payload(
        model_role="trained",
        base_model_registry_key="mask2former",
        base_model_version="3",
        training_task_id="task-123",
        dataset_id=42,
    ))
    assert info_trained.model_role == "trained"
    assert info_trained.base_model_registry_key == "mask2former"
    assert info_trained.dataset_id == 42


def test_register_model_returns_publication_result_and_assigns_alias():
    info = InstanceSegmentationModelInfo.model_validate(_model_payload(
        registry_key="mask2former-ds42-task1",
        model_role="trained",
        dataset_id=42,
    ))
    model = SimpleNamespace(model_info=info, get_artifacts=lambda _tmp_dir: None)

    class FakeMlflowClientWithAlias:
        def __init__(self):
            self.models = {}
            self.aliases = {}

        def search_registered_models(self, filter_string=None):
            return list(self.models.values())

        def get_registered_model(self, name):
            return self.models[name]

        def set_registered_model_tag(self, name, key, value):
            model_obj = self.models.setdefault(name, SimpleNamespace(name=name, tags={}))
            model_obj.tags[key] = value

        def update_registered_model(self, name, description=None):
            self.models[name].description = description

        def set_registered_model_alias(self, name, alias, version):
            self.aliases[(name, alias)] = str(version)

    registry = MLFlowModelRegistry("http://example")
    registry.client = FakeMlflowClientWithAlias()
    run_context = MagicMock()
    run_context.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-99")
    )
    run_context.__exit__.return_value = False

    fake_logged_model = SimpleNamespace(registered_model_version="1")

    with patch("iquana_toolbox.mlflow.mlflow.set_tracking_uri"), \
            patch("iquana_toolbox.mlflow.mlflow.start_run", return_value=run_context), \
            patch("iquana_toolbox.mlflow.mlflow.pyfunc.log_model", return_value=fake_logged_model):
        res = registry.register_model(model, assign_alias="active")

    assert res is not None
    assert res.registry_key == "mask2former-ds42-task1"
    assert res.version == "1"
    assert res.alias == "active"
    assert res.model_uri == "models:/mask2former-ds42-task1/1"
    assert registry.client.aliases.get(("mask2former-ds42-task1", "active")) == "1"


def test_phase7_shared_request_schema_and_canonical_tags():
    """Verify Phase 7 requirements: request fields, string username in trained_by, and boolean task tag."""
    from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
    req = InstanceSegmentationTrainingRequest(
        dataset_id=101,
        image_folder_path="/tmp/images",
        model_registry_key="mask2former",
        user_id="alice_user",
        labels=[],
        annotation_file_url="/tmp/anno.json",
        selected_label_ids=[3, 7, 42],
        enable_hierarchy=True,
    )
    assert req.selected_label_ids == [3, 7, 42]
    assert req.enable_hierarchy is True
    assert req.user_id == "alice_user"

    info = InstanceSegmentationModelInfo(
        registry_key="mask2former-ds101-task777",
        name="Custom Mask2Former",
        description="Phase 7 test",
        usage_tip="Use it",
        model_role="trained",
        dataset_id=req.dataset_id,
        trained_by=req.user_id,
        label_ids=req.selected_label_ids,
        segmentation_mode="hierarchical" if req.enable_hierarchy else "flat",
    )
    tags = MLFlowModelRegistry._model_info_tags(info)
    assert tags["task_instance_segmentation"] == "true"
    assert tags["task"] == "instance-segmentation"
    assert tags["trained_by"] == "alice_user"
    assert tags["dataset_id"] == "101"
    assert tags["label_ids"] == "[3,7,42]"


    parsed = parse_tags_to_model_info(tags)
    assert parsed.trained_by == "alice_user"
    assert parsed.dataset_id == 101
