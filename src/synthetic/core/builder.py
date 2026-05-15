"""StateBuilder — оркестратор. Вызывает блоки в строго определённом порядке.

Порядок критичен: каждый блок зависит от того, что зафиксировали предыдущие.
"""

import random
from core.state import ClinicalCase
from blocks import (
    DiagnosisBlock, TNMBlock, MorphologyBlock,
    MolecularBlock, EcogBlock, RecommendationBlock,
)


class StateBuilder:
    """Pipeline: diagnosis → tnm → morphology → molecular → ecog → recommendations."""

    def __init__(self, schemas: dict, rng: random.Random):
        self.rng = rng
        # Блоки в порядке исполнения. Каждый знает только свой кусок state.
        self.pipeline = [
            DiagnosisBlock(schemas),
            TNMBlock(schemas),
            MorphologyBlock(schemas),
            MolecularBlock(schemas),
            EcogBlock(schemas),
            RecommendationBlock(schemas),
        ]

    def build(self) -> ClinicalCase:
        case = ClinicalCase()
        for block in self.pipeline:
            block.fill(case, self.rng)
        return case
