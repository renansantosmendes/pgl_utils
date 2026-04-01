"""
IBMEC-specific configuration settings
"""


class IBMECConfig:
    """
    Configuration class for IBMEC-specific settings
    """

    INSTITUTION = "IBMEC"
    VERSION = "1.0.0"

    @classmethod
    def get_info(cls):
        """Get IBMEC configuration info"""
        return f"IBMEC Utils v{cls.VERSION}"
