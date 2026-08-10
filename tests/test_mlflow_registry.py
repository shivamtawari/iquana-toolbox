import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas.model_info import ModelInfo


class TestMLFlowModelRegistry(unittest.TestCase):
    def test_register_model_assigns_alias_and_publishes(self):
        registry = MLFlowModelRegistry("http://example")
        registry.client = MagicMock()
        registry.check_registered = MagicMock(return_value=False)

        info = ModelInfo(
            registry_key="demo",
            name="Demo Model",
            description="segmentation model",
            usage_tip="test tip",
            task="instance-segmentation",
            model_role="trained",
            dataset_id=1,
            trained_by="user1",
        )
        model = SimpleNamespace(model_info=info, get_artifacts=lambda p: None)

        fake_logged_model = SimpleNamespace(registered_model_version="7")

        run_context = MagicMock()
        run_context.__enter__.return_value = SimpleNamespace(
            info=SimpleNamespace(run_id="run-123")
        )
        run_context.__exit__.return_value = False
        with patch("iquana_toolbox.mlflow.mlflow.set_tracking_uri") as mock_set_uri, \
                patch("iquana_toolbox.mlflow.mlflow.start_run", return_value=run_context), \
                patch("iquana_toolbox.mlflow.mlflow.pyfunc.log_model", return_value=fake_logged_model):
            res = registry.register_model(model, assign_alias="active")

        self.assertTrue(mock_set_uri.called)
        registry.client.set_registered_model_alias.assert_called_once_with(
            name="demo",
            alias="active",
            version="7",
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.registry_key, "demo")
        self.assertEqual(res.version, "7")
        self.assertEqual(res.alias, "active")

    def test_resolve_base_model_uri_by_alias(self):
        registry = MLFlowModelRegistry("http://example")
        registry.client = MagicMock()
        mv = SimpleNamespace(version="3")
        registry.client.get_model_version_by_alias.return_value = mv

        version, uri = registry.resolve_base_model_uri("mask2former", alias="active")
        self.assertEqual(version, "3")
        self.assertEqual(uri, "models:/mask2former/3")
        registry.client.get_model_version_by_alias.assert_called_once_with(
            name="mask2former", alias="active"
        )


if __name__ == "__main__":
    unittest.main()
