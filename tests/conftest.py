from pathlib import Path

import pytest

from legmri.io import load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "configs" / "study.yaml")
