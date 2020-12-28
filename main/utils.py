import os


def assure_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
