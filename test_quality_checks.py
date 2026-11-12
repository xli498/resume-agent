"""quality_checks.scan_sensitive_info 的确定性测试。

运行：``python -m pytest test_quality_checks.py -v``
或  ：``python -m unittest test_quality_checks -v``

做法是"埋点 + 切片核对"：把敏感片段拼进简历，用 ``text.index(片段)`` 算出
期望位置，逐项比对 ``start``/``end``/``text``，不写死魔数下标。
"""
import copy
import unittest

from quality_checks import scan_sensitive_info


def _expected(rule, text, fragment, search_from=0):
    """按片段在 text 中首次出现的位置构造期望命中。"""
    start = text.index(fragment, search_from)
    return {"rule": rule, "start": start, "end": start + len(fragment),
            "text": fragment}


def mixed_resume():
    """一份简历，每类规则各埋一处，返回 (文本, 各片段)。"""
    fragments = {
        "id_card": "11010519491231002X",       # 17 位数字 + 校验位 X
        "bank_card": "6222021234567890123",   # 19 位连续数字
        "full_address": "广东省深圳市南山区科技园路88号",
    }
    text = "\n".join([
        "姓名：张伟",
        "手机：13800000000",
        "邮箱：zhangwei@example.com",
        f"身份证号：{fragments['id_card']}",
        "教育背景：清华大学 计算机科学与技术 本科",
        f"银行卡：{fragments['bank_card']}",
        "工作经历：某科技有限公司 数据分析师，2020 年 7 月至今",
        f"现住址：{fragments['full_address']} 5 栋 1201",
        "技能：Python、SQL、Tableau",
    ])
    return text, fragments


class MixedResumeTests(unittest.TestCase):
    """一份简历每类规则各埋一处：全部命中且位置准确。"""

    def setUp(self):
        self.resume, self.fragments = mixed_resume()

    def test_all_three_rules_hit_with_accurate_positions(self):
        hits = scan_sensitive_info(self.resume)
        self.assertEqual([h["rule"] for h in hits],
                         ["id_card", "bank_card", "full_address"])

        expected = [
            _expected("id_card", self.resume, self.fragments["id_card"]),
            _expected("bank_card", self.resume, self.fragments["bank_card"]),
            _expected("full_address", self.resume, self.fragments["full_address"]),
        ]
        self.assertEqual(hits, expected)

        # 位置与片段互相自洽：切片回原文必须等于命中片段。
        for hit in hits:
            self.assertEqual(self.resume[hit["start"]:hit["end"]], hit["text"])
        self.assertEqual([h["text"] for h in hits],
                         [self.fragments["id_card"],
                          self.fragments["bank_card"],
                          self.fragments["full_address"]])
        # 每项都带齐四个字段。
        for hit in hits:
            self.assertEqual(list(hit), ["rule", "start", "end", "text"])

    def test_hits_follow_text_order_and_do_not_overlap(self):
        hits = scan_sensitive_info(self.resume)
        starts = [h["start"] for h in hits]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(a["end"] <= b["start"]
                            for a, b in zip(hits, hits[1:])))

    def test_hit_index_is_character_index_not_byte_index(self):
        # 文本里有中文（多字节 UTF-8）；Python 字符下标应能直接切片。
        text = f"身份证：{self.fragments['id_card']}"
        [hit] = scan_sensitive_info(text)
        self.assertEqual(hit["start"], 4)
        self.assertEqual(text[hit["start"]:hit["end"]], hit["text"])


class RuleDetailTests(unittest.TestCase):
    """单条规则的边界：不误报、不重复报。"""

    def test_id_card_with_lowercase_x(self):
        text = "号码：11010519491231002x"
        [hit] = scan_sensitive_info(text)
        self.assertEqual(hit["rule"], "id_card")
        self.assertEqual(hit["text"], "11010519491231002x")

    def test_id_card_is_not_reported_twice_as_bank_card(self):
        text = "身份证：11010519491231002X"
        hits = scan_sensitive_info(text)
        self.assertEqual([h["rule"] for h in hits], ["id_card"])

    def test_bank_card_lengths_16_to_19(self):
        for length in (16, 17, 18, 19):
            number = "".join(str((i * 7 + 3) % 10) for i in range(length))
            # 18 位且末位不是 X 时仍是数字串，应按 bank_card 命中。
            if any(h["rule"] == "id_card" for h in scan_sensitive_info(number)):
                continue
            [hit] = scan_sensitive_info(f"卡号 {number}")
            self.assertEqual((hit["rule"], hit["text"]), ("bank_card", number))

    def test_number_runs_outside_16_to_19_are_ignored(self):
        text = "编号 12345678901234，订单 123456789012345，流水 12345678901234567890"
        self.assertEqual(scan_sensitive_info(text), [])

    def test_address_without_province_is_not_full_address(self):
        self.assertEqual(scan_sensitive_info("住址：北京市朝阳区建国路88号"), [])

    def test_address_requires_house_number(self):
        self.assertEqual(scan_sensitive_info("住址：广东省深圳市南山区科技园路"), [])


class CleanResumeTests(unittest.TestCase):
    """一份干净简历：零命中。"""

    CLEAN = "\n".join([
        "姓名：李娜",
        "手机：139-0000-0000",
        "邮箱：lina@example.com",
        "教育背景：北京大学 经济学 硕士（2018.09-2021.06）",
        "工作经历：",
        "- 2021.07-2023.12 某咨询公司 助理分析师，负责行业研究",
        "- 2024.01-至今 某互联网公司 数据分析师，负责用户增长看板",
        "项目：搭建订单分析看板，覆盖 3 个业务线，月度节省人力 20 小时",
        "技能：Python、SQL、Excel、Tableau",
        "自评：沟通顺畅，交付稳定",
    ])

    def test_clean_resume_has_no_hits(self):
        self.assertEqual(scan_sensitive_info(self.CLEAN), [])

    def test_empty_and_plain_text_have_no_hits(self):
        self.assertEqual(scan_sensitive_info(""), [])
        self.assertEqual(scan_sensitive_info("纯文本，无任何数字与地址。"), [])


class DeterminismTests(unittest.TestCase):
    """同一份简历连续跑三次：结果逐项一致。"""

    def setUp(self):
        self.resume, _ = mixed_resume()

    def _scan(self):
        return scan_sensitive_info(self.resume)

    def test_three_runs_are_item_by_item_identical(self):
        first, second, third = self._scan(), self._scan(), self._scan()
        self.assertTrue(first)  # 埋点确实命中，比较才有意义
        self.assertEqual(len(first), len(second))
        self.assertEqual(len(second), len(third))
        for a, b, c in zip(first, second, third):
            self.assertEqual(a, b)
            self.assertEqual(b, c)
        self.assertEqual(first, second)
        self.assertEqual(second, third)

    def test_input_text_is_not_mutated(self):
        snapshot = copy.deepcopy(self.resume)
        self._scan()
        self.assertEqual(self.resume, snapshot)

    def test_non_string_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            scan_sensitive_info(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
