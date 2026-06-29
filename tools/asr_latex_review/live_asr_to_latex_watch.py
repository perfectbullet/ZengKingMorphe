import re
import json
import sys

out_file = "asr_to_latex_after_20260629.jsonl"
failed_file = "asr_to_latex_failed_live_20260629.log"

log_start_re = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

def is_asr_latex_result_log(entry):
    """
    只保留真正的 ASR→LaTeX 转换结果日志。
    排除：
    ASR→LaTeX decision after classification
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

def extract_duration(payload):
    """
    兼容：
    1. ASR→LaTeX: duration=1.00s, before=..., after=...
    2. ASR→LaTeX after classification: before=..., after=..., duration=1.628s
    """
    matches = re.findall(r"(?:^|[,\s|])duration\s*=\s*([^,\s]+)", payload)
    if not matches:
        return None
    return matches[-1].strip()

def strip_tail_duration(after):
    """
    防止 after 内容里混入尾部 duration。
    例如：
    after=xxx, duration=1.628s
    """
    after = after.strip()

    after = re.sub(
        r"\s*,\s*duration\s*=\s*[^,\s]+\.?\s*$",
        "",
        after,
        flags=re.S
    )

    return clean_value(after)

def parse_asr_latex(entry):
    """
    兼容格式：

    1. ASR→LaTeX: duration=1.00s, before='...', after='...'
    2. ASR→LaTeX: before='...', after='...'
    3. ASR→LaTeX after classification: before=..., after=...
    4. ASR→LaTeX after classification: before=..., after=..., duration=1.628s
    """
    idx = entry.find("ASR→LaTeX")
    if idx < 0:
        return None

    payload = entry[idx:]
    duration = extract_duration(payload)
    log_type = payload.split(":", 1)[0].strip()

    # 旧格式：before='...', after='...'
    m = re.search(
        r"before\s*=\s*'(?P<before>.*?)'\s*,\s*after\s*=\s*'(?P<after>.*?)(?:'\s*)?(?:,\s*duration\s*=\s*[^,\s]+)?\s*$",
        payload,
        re.S
    )
    if m:
        return {
            "type": log_type,
            "duration": duration,
            "before": m.group("before").strip(),
            "after": strip_tail_duration(m.group("after"))
        }

    # 新格式：before=..., after=...
    # 关键：after 后面的尾部 duration 会在 strip_tail_duration() 中剥离
    m = re.search(
        r"before\s*=\s*(?P<before>.*?)\s*,\s*after\s*=\s*(?P<after>.*)$",
        payload,
        re.S
    )
    if m:
        return {
            "type": log_type,
            "duration": duration,
            "before": clean_value(m.group("before")),
            "after": strip_tail_duration(m.group("after"))
        }

    return None

def handle_entry(entry):
    if not is_asr_latex_result_log(entry):
        return

    log_time = entry[:19]
    item = parse_asr_latex(entry)

    if item is None:
        with open(failed_file, "a", encoding="utf-8") as ferr:
            ferr.write(entry + "\n\n")
        print(f"[FAILED] {log_time}", flush=True)
        return

    with open(out_file, "a", encoding="utf-8") as out:
        out.write(json.dumps({
            "time": log_time,
            **item
        }, ensure_ascii=False) + "\n")
        out.flush()

    print(f"[OK] {log_time} duration={item.get('duration')}", flush=True)

buf = []

print("[WATCHING] reading logs from stdin...", flush=True)

for line in sys.stdin:
    if log_start_re.match(line):
        if buf:
            handle_entry("".join(buf).rstrip("\n"))
        buf = [line]
    else:
        if buf:
            buf.append(line)
