from project_name import __version__


def test_example_fixture(example_fixture):
    assert example_fixture == 1


def test_version():
    assert __version__ == "0.1.0"


def test_dataset():
    print("Test your functions here!")
