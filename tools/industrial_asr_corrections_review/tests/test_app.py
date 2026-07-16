import importlib.util
import json
import re
import asyncio
from pathlib import Path

import pytest


APP_PATH = Path(__file__).parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("industrial_review_app", APP_PATH)
review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(review)


def valid_config(**extra):
    config = {
        "version": 1,
        "rules": [{
            "id": "lost_wax_casting", "enabled": True, "canonical": "失蜡铸造",
            "variants": ["湿蜡铸造"], "match_mode": "direct", "priority": 100,
            "unknown_rule": {"kept": True},
        }],
        "unknown_root": "保留",
    }
    config.update(extra)
    return config


def write_config(tmp_path, config=None):
    path = tmp_path / "industrial_asr_corrections.json"
    path.write_text(json.dumps(config or valid_config(), ensure_ascii=False), encoding="utf-8")
    return path


def endpoint(app, path, method):
    for route in app.routes:
        if route.path == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def call(app, path, method, *args):
    return asyncio.run(endpoint(app, path, method)(*args))


def test_load_config_failures_and_chinese(tmp_path):
    path = write_config(tmp_path)
    assert review.load_config(path)["rules"][0]["canonical"] == "失蜡铸造"
    with pytest.raises(ValueError, match="不存在"):
        review.load_config(tmp_path / "missing.json")
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="解析失败"):
        review.load_config(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层"):
        review.load_config(path)
    path.write_text(json.dumps({"version": 1, "rules": {}}), encoding="utf-8")
    assert review.load_config(path)["rules"] == {}


def test_validate_duplicate_ids_variants_and_rule_errors():
    config = valid_config(rules=[
        {"id": "lost_wax_casting", "enabled": True, "canonical": "失蜡铸造", "variants": ["湿蜡"], "match_mode": "direct", "priority": 100},
        {"id": "lost_wax_casting", "enabled": True, "canonical": "失蜡", "variants": ["湿蜡"], "match_mode": "contextual", "priority": 80, "context": {"any": [], "all": [], "none": []}},
        {"id": "wax_tree", "enabled": True, "canonical": "蜡树", "variants": ["蜡术"], "match_mode": "contextual", "priority": 80},
        {"id": "wax_tree", "enabled": True, "canonical": "种蜡树", "variants": ["重蜡树"], "match_mode": "direct", "priority": "80"},
    ])
    result = review.validate_config(config)
    assert not result["ok"]
    assert any(item["rule_id"] == "lost_wax_casting" and item["field"] == "id" for item in result["errors"])
    assert any(item["rule_id"] == "wax_tree" and item["field"] == "id" for item in result["errors"])
    assert any("direct 与 contextual" in item["message"] for item in result["warnings"])
    assert any("variant 出现在多条规则" in item["message"] for item in result["warnings"])
    assert any(item["field"] == "context" for item in result["errors"])
    assert any(item["field"] == "priority" for item in result["errors"])


def test_validate_valid_config_variant_equal_and_unknown_fields():
    assert review.validate_config(valid_config())["ok"]
    broken = valid_config(rules=[{"id": "a", "enabled": True, "canonical": "同词", "variants": ["同词"], "match_mode": "direct", "priority": 1}])
    assert any(item["field"] == "variants" for item in review.validate_config(broken)["errors"])


def test_api_page_config_rules_and_export(tmp_path):
    app = review.create_app(write_config(tmp_path))
    assert "工业实训" in call(app, "/", "GET")
    assert call(app, "/api/config", "GET")["stats"]["rule_count"] == 1
    assert call(app, "/api/rules", "GET")["rules"][0]["canonical"] == "失蜡铸造"
    assert call(app, "/api/rules/{index}", "GET", 0)["rule"]["unknown_rule"] == {"kept": True}
    exported = call(app, "/api/export", "GET")
    assert "失蜡铸造" in exported.body.decode("utf-8")


def test_api_update_create_duplicate_delete_validate_reload(tmp_path):
    path = write_config(tmp_path)
    app = review.create_app(path)
    rule = call(app, "/api/rules/{index}", "GET", 0)["rule"]
    rule["canonical"] = "更新术语"
    assert call(app, "/api/rules/{index}", "PUT", 0, rule)["config"]["rules"][0]["canonical"] == "更新术语"
    created = {"id": "new_rule", "enabled": True, "canonical": "新术语", "variants": ["新错误"], "match_mode": "direct", "priority": 1}
    assert call(app, "/api/rules", "POST", created)["stats"]["rule_count"] == 2
    duplicated = call(app, "/api/rules/{index}/duplicate", "POST", 1)
    assert duplicated["config"]["rules"][2]["id"] == "new_rule_copy"
    assert call(app, "/api/rules/{index}", "DELETE", 2)["stats"]["rule_count"] == 2
    assert call(app, "/api/validate", "POST")["ok"] is True
    path.write_text(json.dumps(valid_config(rules=[]), ensure_ascii=False), encoding="utf-8")
    assert call(app, "/api/reload", "POST")["stats"]["rule_count"] == 0


def test_save_backup_preserves_previous_content_and_unknown_fields(tmp_path):
    path = write_config(tmp_path)
    original = path.read_text(encoding="utf-8")
    config = review.load_config(path)
    config["rules"][0]["canonical"] = "第一次"
    review.save_config(path, config)
    config["rules"][0]["canonical"] = "第二次"
    review.save_config(path, config)
    backups = sorted((tmp_path / "bak_industrial_asr_corrections").glob("*.json"))
    assert len(backups) == 2
    assert re.fullmatch(r"industrial_asr_corrections_\d{8}_\d{6}_\d{6}\.json", backups[0].name)
    assert backups[0].read_text(encoding="utf-8") == original
    saved = path.read_text(encoding="utf-8")
    assert "第二次" in saved and "\\u" not in saved and saved.endswith("\n")
    assert json.loads(saved)["unknown_root"] == "保留"
    assert json.loads(saved)["rules"][0]["unknown_rule"] == {"kept": True}


def test_invalid_or_backup_failure_never_overwrites(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    original = path.read_text(encoding="utf-8")
    invalid = valid_config(rules=[{"id": "", "enabled": True, "canonical": "", "variants": [], "match_mode": "direct", "priority": 1}])
    with pytest.raises(ValueError):
        review.save_config(path, invalid)
    assert path.read_text(encoding="utf-8") == original
    monkeypatch.setattr(review.shutil, "copy2", lambda *_: (_ for _ in ()).throw(OSError("fail")))
    with pytest.raises(OSError):
        review.save_config(path, valid_config())
    assert path.read_text(encoding="utf-8") == original


def test_page_static_contract():
    page = review.PAGE_HTML.lower()
    for required in ("规则", "variants", "context-any", "context-all", "context-none", "/api/", "保存当前规则", "重新加载", "下载 json", "beforeunload"):
        assert required.lower() in page
    assert "login" not in page and "token" not in page and "cdn" not in page and "http://" not in page
