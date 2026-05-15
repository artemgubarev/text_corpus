"""StateBuilder — оркестратор. Вызывает блоки в строго определённом порядке.

Порядок критичен: каждый блок зависит от того, что зафиксировали предыдущие.
"""

import random
from core.state import ClinicalCase
from blocks import (
    DiagnosisBlock, TNMBlock, MorphologyBlock,
    MolecularBlock, EcogBlock,
    SurgeryBlock, RadiotherapyBlock, SystemicTherapyBlock, SupportiveCareBlock,
)


class StateBuilder:
    """Pipeline:
        1. diagnosis → 2. tnm → 3. morphology → 4. molecular → 5. ecog →
        6. surgery → 7. radiotherapy → 8. systemic → 9. supportive

    Блоки рекомендаций идут в конце. supportive — самый последний, потому что
    он смотрит на intent остальных трёх (например, "если куративная цель,
    то post-op реабилитация").
    """

    def __init__(self, schemas: dict, rng: random.Random):
        self.rng = rng
        self.pipeline = [
            DiagnosisBlock(schemas),
            TNMBlock(schemas),
            MorphologyBlock(schemas),
            MolecularBlock(schemas),
            EcogBlock(schemas),
            SurgeryBlock(schemas),
            RadiotherapyBlock(schemas),
            SystemicTherapyBlock(schemas),
            SupportiveCareBlock(schemas),
        ]

    def build(self) -> ClinicalCase:
        case = ClinicalCase()
        for block in self.pipeline:
            block.fill(case, self.rng)
        return case
