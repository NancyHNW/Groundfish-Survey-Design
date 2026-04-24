import os

this_path = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.abspath(os.path.join(this_path, os.pardir, 'data'))
OUT_DIR = os.path.abspath(os.path.join(this_path, os.pardir, 'output'))