"""Package version is single-sourced from installed metadata."""

from importlib.metadata import PackageNotFoundError, version

import laya_serve
from laya_serve.app import create_app
from laya_serve.inference import FakeBackend
from laya_serve.settings import Settings


def test_version_matches_installed_metadata():
    try:
        expected = version("laya-serve")
    except PackageNotFoundError:
        expected = "0.0.0+unknown"
    assert laya_serve.__version__ == expected


def test_openapi_version_matches_package_version():
    app = create_app(Settings(), FakeBackend("laya-english"))
    assert app.version == laya_serve.__version__
