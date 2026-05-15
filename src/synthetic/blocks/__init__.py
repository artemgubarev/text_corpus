from .diagnosis import DiagnosisBlock
from .tnm import TNMBlock
from .morphology import MorphologyBlock
from .molecular import MolecularBlock
from .ecog import EcogBlock
from .surgery import SurgeryBlock
from .radiotherapy import RadiotherapyBlock
from .systemic import SystemicTherapyBlock
from .supportive import SupportiveCareBlock

__all__ = [
    "DiagnosisBlock", "TNMBlock", "MorphologyBlock",
    "MolecularBlock", "EcogBlock",
    "SurgeryBlock", "RadiotherapyBlock", "SystemicTherapyBlock", "SupportiveCareBlock",
]