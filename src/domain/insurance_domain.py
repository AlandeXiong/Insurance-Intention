"""保险行业领域设定 — 常见客户意图参考框架（非固定分类）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class IntentCategory:
    """保险客户常见意图类别 — 供 LLM 参考，非强制枚举。"""
    code: str
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    typical_slots: List[str] = field(default_factory=list)


# 保险行业客户常见意图分类体系（2026 智能客服实践）
INSURANCE_INTENT_CATEGORIES: List[IntentCategory] = [
    IntentCategory(
        code="product_inquiry",
        name="产品咨询",
        description="了解某款保险产品的保障范围、条款内容、特色卖点、适购人群",
        examples=["这款重疾险保哪些疾病", "医疗险包不包含门诊", "适合什么年龄段"],
        typical_slots=["product_name", "insurance_type"],
    ),
    IntentCategory(
        code="premium_inquiry",
        name="保费询价",
        description="询问保费金额、费率、缴费年限、缴费方式",
        examples=["多少钱一年", "30岁男性买50万保额要多少", "年缴还是月缴"],
        typical_slots=["product_name", "age", "gender", "coverage_amount", "payment_period"],
    ),
    IntentCategory(
        code="coverage_terms",
        name="保障条款",
        description="等待期、观察期、免责条款、保障期限、续保规则",
        examples=["等待期多久", "什么情况下不赔", "保证续保吗"],
        typical_slots=["product_name", "clause_type"],
    ),
    IntentCategory(
        code="claims_service",
        name="理赔服务",
        description="理赔流程、所需材料、理赔进度、报销比例",
        examples=["怎么申请理赔", "住院理赔需要什么材料", "理赔多久到账"],
        typical_slots=["product_name", "claim_type", "hospital"],
    ),
    IntentCategory(
        code="purchase",
        name="投保购买",
        description="表达购买意向、询问投保流程、下单操作",
        examples=["我想买一份", "怎么投保", "在线能买吗"],
        typical_slots=["product_name", "budget"],
    ),
    IntentCategory(
        code="product_compare",
        name="产品对比",
        description="对比两款或多款产品的差异",
        examples=["A和B哪个好", "帮我对比一下", "有什么区别"],
        typical_slots=["product_a", "product_b", "compare_dimension"],
    ),
    IntentCategory(
        code="policy_service",
        name="保单服务",
        description="续保、退保、保单变更、查询保单信息",
        examples=["怎么续保", "想退保", "变更受益人", "查我的保单"],
        typical_slots=["policy_no", "service_type"],
    ),
    IntentCategory(
        code="value_added",
        name="权益增值",
        description="增值服务、就医绿通、健康管理、体检权益",
        examples=["有没有绿通服务", "免费体检怎么用"],
        typical_slots=["product_name", "service_type"],
    ),
    IntentCategory(
        code="product_recommend",
        name="产品推荐",
        description="基于用户画像或隐式需求推荐合适产品",
        examples=["经常出差买什么险", "给小孩推荐什么", "有什么适合老年人的"],
        typical_slots=["age", "occupation", "scenario", "budget"],
    ),
    IntentCategory(
        code="complaint_feedback",
        name="投诉建议",
        description="投诉、不满、建议反馈",
        examples=["我要投诉", "服务太差了", "提个建议"],
        typical_slots=["issue_type"],
    ),
    IntentCategory(
        code="greeting_chitchat",
        name="寒暄闲聊",
        description="问候、感谢、无关闲聊",
        examples=["你好", "谢谢", "在吗"],
        typical_slots=[],
    ),
    IntentCategory(
        code="other",
        name="其他",
        description="无法归入以上类别的其他诉求",
        examples=[],
        typical_slots=[],
    ),
]

CATEGORY_BY_CODE: Dict[str, IntentCategory] = {c.code: c for c in INSURANCE_INTENT_CATEGORIES}

# 产品实体库 — 用于指代消解与槽位抽取
PRODUCT_ENTITIES: Dict[str, List[str]] = {
    "安心保重疾险2026": ["安心保", "安心保重疾", "这款重疾", "它", "这个"],
    "康乐医疗险Plus": ["康乐医疗", "康乐Plus", "医疗险", "这款医疗"],
    "畅行意外险": ["畅行意外", "意外险", "出差险"],
}

# 通用槽位定义（跨意图共享，由 LLM 动态填充）
COMMON_SLOT_HINTS = [
    "product_name", "product_a", "product_b",
    "age", "gender", "coverage_amount", "payment_period", "budget",
    "insurance_type", "claim_type", "policy_no", "occupation", "scenario",
]


def build_category_prompt() -> str:
    """构建供 LLM 参考的意图分类说明文本。"""
    lines = ["以下是保险行业客户常见意图分类（仅供参考，可灵活细化）：\n"]
    for cat in INSURANCE_INTENT_CATEGORIES:
        lines.append(f"- 【{cat.name}】({cat.code}): {cat.description}")
        if cat.examples:
            lines.append(f"  示例: {'; '.join(cat.examples[:3])}")
        if cat.typical_slots:
            lines.append(f"  常见槽位: {', '.join(cat.typical_slots)}")
    return "\n".join(lines)


def get_category_name(code: str) -> str:
    cat = CATEGORY_BY_CODE.get(code)
    return cat.name if cat else code
