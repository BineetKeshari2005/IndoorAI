"""Structural validation tests for IndoorAI project scaffolding."""

from pathlib import Path
import unittest

from src.core.config import get_settings
from src.core.logging import setup_logger


class TestProjectScaffolding(unittest.TestCase):
    """Test suite ensuring project structure integrity and core initialization."""

    def setUp(self) -> None:
        self.settings = get_settings()
        self.root_dir = self.settings.paths.root_dir

    def test_required_directories_exist(self) -> None:
        """Verify that all architectural directories are present."""
        expected_dirs = [
            "data/raw",
            "data/processed",
            "data/external",
            "notebooks",
            "src/data",
            "src/preprocessing",
            "src/features",
            "src/models/traditional_ml",
            "src/models/deep_learning",
            "src/models/vision",
            "src/models/multimodal",
            "src/training",
            "src/evaluation",
            "src/explainability",
            "src/sensor_fusion",
            "src/mapping",
            "backend/app/api",
            "backend/app/services",
            "backend/app/schemas",
            "backend/app/core",
            "android",
            "frontend",
            "experiments",
            "tests",
            "configs",
            "docs",
        ]
        for rel_dir in expected_dirs:
            target_path = self.root_dir / rel_dir
            self.assertTrue(
                target_path.exists() and target_path.is_dir(),
                f"Missing expected directory: {rel_dir} (resolved to {target_path})",
            )

    def test_required_files_exist(self) -> None:
        """Verify foundational files exist."""
        expected_files = [
            "pyproject.toml",
            ".gitignore",
            ".env.example",
            "README.md",
            "IMPLEMENTATION_STATUS.md",
            "configs/base_config.yaml",
            "configs/logging_config.yaml",
            "docs/SYSTEM_ARCHITECTURE.md",
            "docs/ML_ARCHITECTURE.md",
            "docs/DATA_STRATEGY.md",
            "docs/DEVELOPMENT_ROADMAP.md",
            "docs/PRIVACY_AND_SECURITY.md",
        ]
        for rel_file in expected_files:
            target_path = self.root_dir / rel_file
            self.assertTrue(
                target_path.exists() and target_path.is_file(),
                f"Missing expected file: {rel_file}",
            )

    def test_configuration_loading(self) -> None:
        """Verify that settings can be loaded without hardcoded failures."""
        self.assertEqual(self.settings.app_name, "IndoorAI")
        self.assertTrue(self.settings.paths.root_dir.exists())
        self.assertTrue(self.settings.paths.data_dir.exists())

    def test_logger_initialization(self) -> None:
        """Verify standard logger setup."""
        logger = setup_logger("TestLogger")
        self.assertIsNotNone(logger)
        logger.info("Structural test logger initialized successfully.")


if __name__ == "__main__":
    unittest.main()

