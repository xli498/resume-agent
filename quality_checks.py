"""确定性敏感信息扫描（纯规则，不调用模型）。

用于在把简历文本送去改写/导出/展示之前，先找出其中可能不该外露的
个人敏感信息。设计原则：

- 纯函数：同样的输入永远返回逐项相同的输出（无随机、无时间、无 IO）；
- 只"识别"位置和片段，不脱敏、不修改输入文本；
- 所有规则都是确定性正则，不调用任何模型或外部服务。

规则：

``id_card``      18 位身份证号：17 位数字 + 校验位 ``[0-9Xx]``。
``bank_card``    16–19 位连续数字串（银行卡样式），且不能是身份证号
                 本身（与之完全重叠的数字串不再重复报 bank_card）。
``full_address`` 完整住址：省 + 市 + 区/县 + 路/街/道 + 号，从省名
                 开始、到门牌号结尾整体命中。

命中的数字规则前后不能紧贴其他字母或数字，因此更长的串不会被截取，
短于 16 位的数字串（手机号、年份、工号等）也不会命中。

用法::

    from quality_checks import scan_sensitive_info

    for hit in scan_sensitive_info(resume_text):
        print(hit["rule"], hit["start"], hit["end"], hit["text"])
"""
from __future__ import annotations

import re
from typing import TypedDict


class SensitiveHit(TypedDict):
    """一条命中记录。

    - ``rule``：规则名（``id_card`` / ``bank_card`` / ``full_address``）；
    - ``start``：命中片段在原文中的起始下标（含，Python 字符下标）；
    - ``end``：命中片段在原文中的结束下标（不含），即 ``text == s[start:end]``；
    - ``text``：命中的原文片段。
    """

    rule: str
    start: int
    end: int
    text: str


# 数字/字母边界：命中的数字串两侧不能紧邻字母或数字，避免从更长的串中截取。
_LEFT_BOUNDARY = r"(?<![0-9A-Za-z])"
_RIGHT_BOUNDARY = r"(?![0-9A-Za-z])"

# 18 位身份证号：17 位数字 + 校验位 [0-9Xx]。
ID_CARD_PATTERN = re.compile(_LEFT_BOUNDARY + r"[0-9]{17}[0-9Xx]" + _RIGHT_BOUNDARY)

# 银行卡样式：16–19 位连续数字，贪婪取最长一段。
BANK_CARD_PATTERN = re.compile(_LEFT_BOUNDARY + r"[0-9]{16,19}" + _RIGHT_BOUNDARY)

# 完整住址：省 + 市 + 区/县 + 路/街/道 + 号。各级名称只做"形态"匹配
# （任意汉字名 + 该级后缀），不依赖行政区划词表，保证纯确定性。
_ADDR_GAP = r"[\s,，、·\-]{0,3}"
FULL_ADDRESS_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{2,8}省"
    + _ADDR_GAP
    + r"[\u4e00-\u9fff]{2,10}市"
    + _ADDR_GAP
    + r"[\u4e00-\u9fff]{2,10}(?:区|县)"
    + _ADDR_GAP
    + r"[\u4e00-\u9fff]{2,20}(?:路|街|道)"
    + _ADDR_GAP
    + r"[0-9]{1,5}号"
)

# (规则名, 编译后的正则)。顺序固定，保证同一输入的输出顺序稳定。
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("id_card", ID_CARD_PATTERN),
    ("bank_card", BANK_CARD_PATTERN),
    ("full_address", FULL_ADDRESS_PATTERN),
)


def scan_sensitive_info(text: str) -> list[SensitiveHit]:
    """扫描简历文本中的敏感信息，返回命中列表。

    纯函数：不修改 ``text``，不依赖任何外部状态；同一输入每次返回逐项
    相同的结果（规则顺序固定，规则内按出现位置升序）。

    ``start``/``end`` 是 Python 字符下标，满足
    ``hit["text"] == text[hit["start"]:hit["end"]]``。

    非字符串输入（如 ``None``）抛 ``TypeError``。
    """
    if not isinstance(text, str):
        raise TypeError(f"scan_sensitive_info 需要 str，收到 {type(text).__name__}")

    id_spans = [m.span() for m in ID_CARD_PATTERN.finditer(text)]

    hits: list[SensitiveHit] = []
    for rule, pattern in _RULES:
        for match in pattern.finditer(text):
            span = match.span()
            # 身份证号本身也是 16–19 位数字串：只按 id_card 报一次，避免重复。
            if rule == "bank_card" and span in id_spans:
                continue
            hits.append(
                {
                    "rule": rule,
                    "start": span[0],
                    "end": span[1],
                    "text": match.group(0),
                }
            )
    return hits
