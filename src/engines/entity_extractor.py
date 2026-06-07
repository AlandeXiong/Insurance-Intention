"""实体抽取器 — 轻量辅助，不做固定意图分类。"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from src.domain.insurance_domain import CATEGORY_BY_CODE, PRODUCT_ENTITIES
from src.models.intent import SessionContext, Slot, SlotStatus

# 关键词 → 参考分类 hint（降级路径用，非强制分类）
CATEGORY_HINTS: list[tuple[list[str], str, str]] = [
    (["等待期", "观察期", "免责", "保障期限"], "coverage_terms", "查询保障条款"),
    (["保费", "多少钱", "价格", "费率", "年缴"], "premium_inquiry", "咨询保费"),
    (["理赔", "报销", "赔付"], "claims_service", "咨询理赔相关事宜"),
    (["对比", "比较", "哪个好", "区别"], "product_compare", "对比保险产品"),
    (["购买", "投保", "买一份", "怎么买"], "purchase", "表达投保意向"),
    (["续保", "退保", "保单", "变更"], "policy_service", "办理保单服务"),
    (["推荐", "买什么", "适合"], "product_recommend", "寻求产品推荐"),
    (["出差", "旅行", "意外"], "product_recommend", "基于场景的产品推荐"),
    (["你好", "您好", "在吗"], "greeting_chitchat", "寒暄问候"),
    (["重疾", "医疗", "保障范围", "保什么"], "product_inquiry", "咨询产品保障"),
]


class EntityExtractor:
    """抽取实体槽位 + 降级时的意图 hint，不替代 LLM 动态识别。"""

    SLOT_PATTERNS = {
        "age": re.compile(r"(\d{1,3})\s*岁"),
        "coverage_amount": re.compile(r"(\d+)\s*万"),
        "payment_period": re.compile(r"(?<![0-9])(\d{1,2})\s*年(?:交|缴|期)?"),
        "budget": re.compile(r"预算\s*(\d+)\s*元?"),
    }

    def infer_intent_hint(
        self, utterance: str, ctx: Optional[SessionContext] = None
    ) -> Tuple[str, str, float]:
        """降级路径：返回 (intent_label, category_code, confidence)。"""
        best_label = "未能识别用户意图"
        best_cat = "other"
        best_conf = 0.3

        for keywords, cat, label in CATEGORY_HINTS:
            hits = sum(1 for kw in keywords if kw in utterance)
            if hits > 0:
                conf = min(0.65 + 0.05 * hits, 0.85)
                if conf > best_conf:
                    best_conf = conf
                    best_cat = cat
                    best_label = label

        if ctx and ctx.active_intent_label and len(utterance) < 15:
            return ctx.active_intent_label, ctx.active_category or "other", 0.6

        return best_label, best_cat, best_conf

    def extract_entities(
        self, utterance: str, ctx: Optional[SessionContext] = None
    ) -> Dict[str, Slot]:
        slots: Dict[str, Slot] = {}

        product = self._match_product(utterance, ctx)
        if product:
            slots["product_name"] = Slot(
                name="product_name", value=product, status=SlotStatus.FILLED, confidence=0.85
            )

        for name, pattern in self.SLOT_PATTERNS.items():
            m = pattern.search(utterance)
            if m:
                slots[name] = Slot(
                    name=name,
                    value=m.group(1) if m.lastindex else m.group(0),
                    status=SlotStatus.FILLED,
                    confidence=0.85,
                )

        if any(k in utterance for k in ("经常出差", "经常旅行")):
            slots["scenario"] = Slot(name="scenario", value="高频出行", status=SlotStatus.FILLED, confidence=0.8)

        if ctx:
            for name, mem in ctx.slot_memory.items():
                if name not in slots and mem.value:
                    slots[name] = Slot(
                        name=name,
                        value=mem.value,
                        status=SlotStatus.INHERITED,
                        confidence=mem.confidence * 0.9,
                        source_turn=mem.source_turn,
                    )

        return slots

    def extract_compare_products(self, utterance: str) -> Dict[str, Slot]:
        found: list[str] = []
        for product, aliases in PRODUCT_ENTITIES.items():
            if product in utterance or any(a in utterance for a in aliases if len(a) > 1):
                if product not in found:
                    found.append(product)
        slots: Dict[str, Slot] = {}
        if len(found) >= 1:
            slots["product_a"] = Slot(name="product_a", value=found[0], status=SlotStatus.FILLED, confidence=0.88)
        if len(found) >= 2:
            slots["product_b"] = Slot(name="product_b", value=found[1], status=SlotStatus.FILLED, confidence=0.88)
        return slots

    def _match_product(self, utterance: str, ctx: Optional[SessionContext]) -> Optional[str]:
        for product, aliases in PRODUCT_ENTITIES.items():
            if product in utterance:
                return product
            for alias in aliases:
                if len(alias) > 1 and alias in utterance:
                    return product
        if ctx and ctx.active_product:
            return ctx.active_product
        return None
