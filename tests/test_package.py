from importlib import import_module


def test_package_can_be_imported() -> None:
    # Given: the project has been installed in editable mode.
    package_name = "minigpt"

    # When: Python imports the public package.
    package = import_module(package_name)

    # Then: the package exposes its version metadata.
    assert package.__version__ == "0.1.0"
