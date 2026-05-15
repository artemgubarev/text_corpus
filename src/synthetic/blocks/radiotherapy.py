"""Блок рекомендации: Лучевая терапия."""

from ._base_recommendation import BaseRecommendationBlock


class RadiotherapyBlock(BaseRecommendationBlock):
    SCHEMA_KEY = "radiotherapy"
    STATE_FIELDS = ("radiotherapy_rule_id", "radiotherapy_intent", "radiotherapy_text")
