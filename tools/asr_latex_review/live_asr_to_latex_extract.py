import re
import json

log_file = "/data/metahuman_work/ZengKingMorphe/ai-service/logs/ai_service.log"
out_file = "asr_to_latex_after_20260629.jsonl"
failed_file = "asr_to_latex_failed_20260629.log"
start_time = "2026-06-28 09:00:00"

log_start_re = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

def iter_log_entries(f):
    """将多行日志合并成一条日志记录。"""
    buf = []
    for line in f:
        if log_start_re.match(line):
            if buf:
                yield "".join(buf).rstrip("\n")
            buf = [line]
        else:
            if buf:
                buf.append(line)

    if buf:
        yield "".join(buf).rstrip("\n")

def is_asr_latex_result_log(entry):
    """
    只保留真正的 ASR→LaTeX 转换结果日志。
    排除 ASR→LaTeX decision after classification 这种决策日志。
    """
    return (
        "ASR→LaTeX:" in entry
        or "ASR→LaTeX after classification:" in entry
    )

def clean_value(s):
    s = s.strip()

    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].strip()

    if s.startswith("'"):
        s = s[1:]

    if s.endswith("'"):
        s = s[:-1]

    return s.strip()

def parse_asr_latex(entry):
    """
    兼容格式：

    1. ASR→LaTeX: duration=1.00s, before='...', after='...'
    2. ASR→LaTeX: before='...', after='...'
    3. ASR→LaTeX after classification: before=..., after=...
    """
    idx = entry.find("ASR→LaTeX")
    if idx < 0:
        return None

    payload = entry[idx:]

    duration = None
    m_duration = re.search(r"duration=([^,]+)", payload)
    if m_duration:
        duration = m_duration.group(1).strip()

    log_type = payload.split(":", 1)[0].strip()

    # 旧格式：before='...', after='...'
    m = re.search(
        r"before\s*=\s*'(?P<before>.*?)'\s*,\s*after\s*=\s*'(?P<after>.*?)(?:'\s*)?$",
        payload,
        re.S
    )
    if m:
        return {
            "type": log_type,
            "duration": duration,
            "before": m.group("before").strip(),
            "after": m.group("after").strip()
        }

    # 新格式：before=..., after=...
    m = re.search(
        r"before\s*=\s*(?P<before>.*?)\s*,\s*after\s*=\s*(?P<after>.*?)"
        r"(?:,\s*duration=[^,\n]+)?$",
        payload,
        re.S
    )
    if m:
        return {
            "type": log_type,
            "duration": duration,
            "before": clean_value(m.group("before")),
            "after": clean_value(m.group("after"))
        }

    return None

total_asr_latex_result = 0
after_time_filter = 0
parsed = 0
failed = 0

with open(log_file, "r", encoding="utf-8", errors="ignore") as f, \
     open(out_file, "w", encoding="utf-8") as out, \
     open(failed_file, "w", encoding="utf-8") as ferr:

    for entry in iter_log_entries(f):
        if not is_asr_latex_result_log(entry):
            continue

        total_asr_latex_result += 1

        log_time = entry[:19]
        if log_time < start_time:
            continue

        after_time_filter += 1

        item = parse_asr_latex(entry)
        if item is None:
            failed += 1
            ferr.write(entry)
            ferr.write("\n\n")
            continue

        out.write(json.dumps({
            "time": log_time,
            **item
        }, ensure_ascii=False) + "\n")

        parsed += 1

print(f"total_asr_latex_result={total_asr_latex_result}")
print(f"after_time_filter={after_time_filter}")
print(f"parsed={parsed}")
print(f"failed={failed}")
print(f"output={out_file}")
print(f"failed_output={failed_file}")
