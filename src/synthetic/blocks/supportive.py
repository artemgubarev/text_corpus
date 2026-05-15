"""Блок рекомендации: Поддерживающая терапия и наблюдение."""

from ._base_recommendation import BaseRecommendationBlock


class SupportiveCareBlock(BaseRecommendationBlock):
    SCHEMA_KEY = "supportive_care"
    STATE_FIELDS = ("supportive_rule_id", "supportive_intent", "supportive_text")
