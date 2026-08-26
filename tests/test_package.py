import minigpt
from minigpt._version import __version__


def test_package_can_be_imported() -> None:
    # Given: the project has been installed in editable mode.
    # When: Python imports the public package.
    package = minigpt

    # Then: the package exposes the single authored version value.
    assert package.__version__ == __version__
