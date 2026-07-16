"""工训术语确定性归一化测试。"""

import json

import pytest

from app.services.conversation import industrial_term_normalizer as normalizer


@pytest.fixture(autouse=True)
def clear_rule_cache(monkeypatch):
    monkeypatch.delenv("INDUSTRIAL_TERM_NORMALIZATION_ENABLED", raising=False)
    yield


def _set_rules(monkeypatch, tmp_path, rules, *, version=1):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"version": version, "rules": rules}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(normalizer, "RULES_PATH", path)
    return path


def test_lost_wax_casting_core_case():
    result = normalizer.normalize_industrial_terms(
        "简述石膏灌浆在湿蜡铸造中的作用以及需要注意的关键点。"
    )

    assert result.normalized == "简述石膏灌浆在失蜡铸造中的作用以及需要注意的关键点。"
    assert result.applied is True
    assert [(item.source, item.target) for item in result.matches] == [
        ("湿蜡铸造", "失蜡铸造")
    ]


def test_correct_term_is_unchanged():
    result = normalizer.normalize_industrial_terms("失蜡铸造需要经过脱蜡和焙烧。")

    assert result.applied is False
    assert result.matches == ()
    assert result.normalized == result.original


def test_longest_variant_wins_over_overlapping_short_variant():
    result = normalizer.normalize_industrial_terms("湿蜡铸造有哪些主要工序？")

    assert result.normalized == "失蜡铸造有哪些主要工序？"
    assert [(item.source, item.target) for item in result.matches] == [
        ("湿蜡铸造", "失蜡铸造")
    ]


def test_replacements_are_non_cascading(monkeypatch, tmp_path):
    _set_rules(
        monkeypatch,
        tmp_path,
        [
            {"id": "first", "enabled": True, "canonical": "B", "variants": ["A"], "match_mode": "direct", "priority": 1},
            {"id": "second", "enabled": True, "canonical": "C", "variants": ["B"], "match_mode": "direct", "priority": 1},
        ],
    )

    result = normalizer.normalize_industrial_terms("A")

    assert result.normalized == "B"
    assert [item.rule_id for item in result.matches] == ["first"]


def test_multiple_non_overlapping_and_direct_rules():
    result = normalizer.normalize_industrial_terms("用油标卡尺检查手饰，再用精工锉进行直磨。")

    assert result.normalized == "用游标卡尺检查首饰，再用金工锉进行执模。"
    assert [item.source for item in result.matches] == ["油标卡尺", "手饰", "精工锉", "直磨"]


def test_contextual_positive_and_negative_cases():
    assert normalizer.normalize_industrial_terms("发廊釉料烧制后为什么会产生气泡？").normalized == "珐琅釉料烧制后为什么会产生气泡？"
    assert normalizer.normalize_industrial_terms("金属首饰浇筑前为什么要制作蜡模？").normalized == "金属首饰浇铸前为什么要制作蜡模？"
    assert normalizer.normalize_industrial_terms("珐琅烧焰时要注意什么？").normalized == "珐琅烧艳时要注意什么？"
    assert normalizer.normalize_industrial_terms("蜡模铸造时重蜡如何处理？").normalized == "蜡模铸造时种蜡如何处理？"
    assert normalizer.normalize_industrial_terms("首饰蜡模铸造中的蜡术如何组装？").normalized == "首饰蜡模铸造中的蜡树如何组装？"
    for query in (
        "法国以前使用法郎作为货币。",
        "附近有哪些发廊？",
        "这家发廊可以剪发吗？",
        "混凝土浇筑有哪些注意事项？",
        "楼板浇筑完成后多久可以拆模？",
        "湿蜡是一种什么材料？",
    ):
        result = normalizer.normalize_industrial_terms(query)
        assert result.normalized == query
        assert result.applied is False


def test_missing_and_invalid_json_leave_query_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(normalizer, "RULES_PATH", tmp_path / "missing.json")
    assert normalizer.normalize_industrial_terms("湿蜡铸造").applied is False

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(normalizer, "RULES_PATH", invalid_path)
    assert normalizer.normalize_industrial_terms("湿蜡铸造").matches == ()


def test_invalid_and_duplicate_rules_do_not_block_valid_rule(monkeypatch, tmp_path):
    _set_rules(
        monkeypatch,
        tmp_path,
        [
            {"id": "valid", "enabled": True, "canonical": "标准", "variants": ["错误"], "match_mode": "direct", "priority": 1},
            {"id": "valid", "enabled": True, "canonical": "其他", "variants": ["重复"], "match_mode": "direct", "priority": 1},
            {"id": "broken", "enabled": True, "canonical": "", "variants": ["坏"], "match_mode": "direct", "priority": 1},
        ],
    )
    warnings = []
    monkeypatch.setattr(normalizer.logger, "warning", warnings.append)

    result = normalizer.normalize_industrial_terms("错误重复坏")

    assert result.normalized == "标准重复坏"
    assert any("Invalid industrial term rule skipped" in message for message in warnings)


def test_environment_switch_disables_all_changes(monkeypatch):
    monkeypatch.setenv("INDUSTRIAL_TERM_NORMALIZATION_ENABLED", "false")

    result = normalizer.normalize_industrial_terms("湿蜡铸造")

    assert result.normalized == "湿蜡铸造"
    assert result.applied is False


def test_rules_are_reloaded_from_disk_for_each_call(monkeypatch, tmp_path):
    path = _set_rules(
        monkeypatch,
        tmp_path,
        [{"id": "term", "enabled": True, "canonical": "标准一", "variants": ["错误"], "match_mode": "direct", "priority": 1}],
    )
    assert normalizer.normalize_industrial_terms("错误").normalized == "标准一"

    path.write_text(
        json.dumps(
            {"version": 1, "rules": [{"id": "term", "enabled": True, "canonical": "标准二", "variants": ["错误"], "match_mode": "direct", "priority": 1}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert normalizer.normalize_industrial_terms("错误").normalized == "标准二"
