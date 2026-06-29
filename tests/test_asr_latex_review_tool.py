import asyncio
import json
from pathlib import Path

import httpx
import pytest

from tools.asr_latex_review.app import (
    build_stats,
    create_app,
    filter_records,
    load_records,
    save_records_atomic,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_records_adds_defaults_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"before": "口语输入", "after": "公式"}\n\n  \n'
        '{"time": "now"}\n',
        encoding="utf-8",
    )

    records = load_records(path)

    assert records == [
        {"id": None, "before": "口语输入", "after": "公式", "review_status": None},
        {"time": "now", "id": None, "before": "", "after": "", "review_status": None},
    ]


def test_load_records_adds_id_none_and_preserves_existing_id(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    write_jsonl(
        path,
        [
            {"before": "原文", "after": "转换"},
            {"id": "MATH-001", "before": "原文", "after": "转换"},
        ],
    )

    records = load_records(path)

    assert records[0]["id"] is None
    assert records[1]["id"] == "MATH-001"


def test_load_records_reports_invalid_json_line_number(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"before": "ok"}\n\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="第 3 行"):
        load_records(path)


def test_save_records_creates_backup_and_preserves_it(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    original = '{"before": "原始内容"}\n'
    path.write_text(original, encoding="utf-8")

    save_records_atomic(path, [{"before": "第一次保存", "review_status": None}])
    backup_path = Path(f"{path}.bak")
    assert backup_path.read_text(encoding="utf-8") == original

    save_records_atomic(path, [{"before": "第二次保存", "review_status": "correct"}])
    assert backup_path.read_text(encoding="utf-8") == original


def test_save_records_omits_index_and_preserves_chinese(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    write_jsonl(path, [{"before": "旧值"}])

    save_records_atomic(
        path,
        [{"index": 7, "before": "中文输入", "after": "$x^{2}$", "review_status": None}],
    )

    saved_text = path.read_text(encoding="utf-8")
    saved_record = json.loads(saved_text)
    assert "中文输入" in saved_text
    assert "\\u4e2d" not in saved_text
    assert "index" not in saved_record
    assert saved_text.count("\n") == 1


def test_save_records_writes_id_but_not_index(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    write_jsonl(path, [{"before": "原文", "after": "转换"}])
    records = load_records(path)
    records[0]["id"] = "MATH-001"
    records[0]["index"] = 123

    save_records_atomic(path, records)

    saved_record = json.loads(path.read_text(encoding="utf-8"))
    assert saved_record["id"] == "MATH-001"
    assert "index" not in saved_record


def test_api_updates_normalizes_and_preserves_id(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    write_jsonl(
        path,
        [{"id": "OLD-ID", "before": "原文", "after": "转换"}],
    )

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=create_app(path))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            loaded = await client.get("/api/records?status=all")
            assert loaded.status_code == 200
            assert loaded.json()["records"][0]["id"] == "OLD-ID"

            updated = await client.post(
                "/api/records/0",
                json={
                    "id": "  MATH-001  ",
                    "after": "新转换",
                    "review_status": "correct",
                },
            )
            assert updated.status_code == 200
            assert updated.json()["record"]["id"] == "MATH-001"

            preserved = await client.post(
                "/api/records/0",
                json={"after": "再次转换", "review_status": "correct"},
            )
            assert preserved.status_code == 200
            assert preserved.json()["record"]["id"] == "MATH-001"

            cleared = await client.post(
                "/api/records/0",
                json={"id": "   ", "after": "再次转换", "review_status": None},
            )
            assert cleared.status_code == 200
            assert cleared.json()["record"]["id"] is None

    asyncio.run(exercise_api())

    saved_record = json.loads(path.read_text(encoding="utf-8"))
    assert saved_record["id"] is None
    assert "index" not in saved_record


def test_filter_records_supports_all_statuses() -> None:
    records = [
        {"review_status": None, "value": 1},
        {"review_status": "correct", "value": 2},
        {"review_status": "incorrect", "value": 3},
    ]

    assert filter_records(records, "all") == records
    assert [record["value"] for record in filter_records(records, "unreviewed")] == [1]
    assert [record["value"] for record in filter_records(records, "correct")] == [2]
    assert [record["value"] for record in filter_records(records, "incorrect")] == [3]

    with pytest.raises(ValueError, match="不支持的过滤状态"):
        filter_records(records, "invalid")


def test_build_stats() -> None:
    records = [
        {"review_status": None},
        {"review_status": "correct"},
        {"review_status": "correct"},
        {"review_status": "incorrect"},
    ]

    assert build_stats(records) == {
        "total": 4,
        "unreviewed": 1,
        "correct": 2,
        "incorrect": 1,
    }
