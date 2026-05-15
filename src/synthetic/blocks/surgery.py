"""Блок рекомендации: Хирургическое лечение."""

from ._base_recommendation import BaseRecommendationBlock


class SurgeryBlock(BaseRecommendationBlock):
    SCHEMA_KEY = "surgery"
    STATE_FIELDS = ("surgery_rule_id", "surgery_intent", "surgery_text")
