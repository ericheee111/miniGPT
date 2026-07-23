import minigpt


def test_package_can_be_imported() -> None:
    # Given: the project has been installed in editable mode.
    # When: Python imports the public package.
    package = minigpt

    # Then: the package exposes its version metadata.
    assert package.__version__ == "0.1.0"
