"""
Basic example of using post_graduation_utils
"""

from post_graduation_utils import core
from post_graduation_utils.ml import preprocessing, models
from post_graduation_utils.puc import config as puc_config
from post_graduation_utils.ibmec import config as ibmec_config


def main():
    """Run basic examples"""
    
    print("=" * 50)
    print("Post Graduation Utils - Basic Example")
    print("=" * 50)
    
    # Core utilities
    print("\n1. Core Utilities:")
    print(f"   {core.utils.placeholder()}")
    
    # ML utilities
    print("\n2. Machine Learning Utilities:")
    print(f"   {models.placeholder()}")
    print(f"   {preprocessing.placeholder()}")
    
    # Institution-specific
    print("\n3. Institution-Specific Tools:")
    print(f"   PUC: {puc_config.PUCConfig.get_info()}")
    print(f"   IBMEC: {ibmec_config.IBMECConfig.get_info()}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
