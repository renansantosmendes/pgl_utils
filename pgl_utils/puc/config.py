"""
PUC-specific configuration settings
"""


class PUCConfig:
    """
    Configuration class for PUC-specific settings
    """

    INSTITUTION = "PUC"
    VERSION = "1.0.0"

    @classmethod
    def get_info(cls):
        """Get PUC configuration info"""
        return f"PUC Utils v{cls.VERSION}"
