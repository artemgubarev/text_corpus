"""Блок рекомендации: Системная терапия."""

from ._base_recommendation import BaseRecommendationBlock


class SystemicTherapyBlock(BaseRecommendationBlock):
    SCHEMA_KEY = "systemic_therapy"
    STATE_FIELDS = ("systemic_rule_id", "systemic_intent", "systemic_text")
