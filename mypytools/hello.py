"""
Learning how to build a python package
"""


def say_hello():
    """
    Say hello
    """
    print(f"Hello, I am a happy hello-saying function in {__file__}")


if __name__ == "__main__":
    print(__file__)
    print(__doc__)
    print(f"I am doint something in {__name__}")
