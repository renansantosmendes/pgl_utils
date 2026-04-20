"""
Basic example of using pgl_utils
"""

import sys, os

os.environ['TF_CPP_MAX_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

from pgl_utils import core
from pgl_utils.ml import preprocessing, models


def main():
    """Run basic examples"""

    print("=" * 50)
    print("Post Graduation Utils - Basic Example")
    print("=" * 50)


    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
