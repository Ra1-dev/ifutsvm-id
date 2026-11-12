"""Keep test temporaries inside the workspace, isolated across Python runs."""
from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    if config.option.basetemp is None:
        parent = Path(config.rootpath) / 'tmp' / 'pytest'
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid4().hex)
