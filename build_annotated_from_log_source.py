#!/usr/bin/env python3
"""
Build annotated JSON from source files:
  - JSON log_source  (Algorithm-Generated / automatic): fetch JSON, merge annotations
  - TXT log_source   (MAST console_log.txt / manual)  : fetch console_log.txt +
    prompt.txt + expected_answer.txt, parse turns, merge annotations
"""
import argparse
import json
import re
import requests
from pathlib import Path
from urllib.parse import urlparse

_AGENT_DELIMITER = re.compile(r"^-{10} (.+?) -{10}$")


# ── URL helpers ──────────────────────────────────────────────────────────────

def github_tree_to_raw(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return url
    parts = parsed.path.strip("/").split("/")
    # /<owner>/<repo>/tree/<branch>/<path...>
    if len(parts) >= 5 and parts[2] == "tree":
        owner, repo, _, branch = parts[:4]
        file_path = "/".join(parts[4:])
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    return url


def fetch_text(url: str, timeout: int = 60) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# ── console_log.txt parser ───────────────────────────────────────────────────

def parse_console_log(content: str) -> list[dict]:
    """
    Split console_log.txt into a list of {role, content} dicts.

    Parsing rules:
      - Skip everything before 'SCENARIO.PY STARTING !#!#'
      - Lines matching '---------- AGENT_NAME ----------' start a new turn
      - All text until the next delimiter belongs to the current turn
    """
    marker = "SCENARIO.PY STARTING !#!#"
    pos = content.find(marker)
    if pos != -1:
        content = content[pos + len(marker):]

    turns: list[dict] = []
    current_agent: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        m = _AGENT_DELIMITER.match(line.strip())
        if m:
            if current_agent is not None:
                turns.append({
                    "role": current_agent,
                    "content": "\n".join(current_lines).strip(),
                })
            current_agent = m.group(1).strip()
            current_lines = []
        else:
            if current_agent is not None:
                current_lines.append(line)

    if current_agent is not None:
        turns.append({
            "role": current_agent,
            "content": "\n".join(current_lines).strip(),
        })

    return turns


# ── transform functions ───────────────────────────────────────────────────────

def _apply_annotations(history: list[dict], annotations: list[dict]) -> list[dict]:
    annotation_by_step = {item["step"]: item for item in annotations}
    for item in history:
        ann = annotation_by_step.get(item["step"], {})
        item["fail_annotation"] = ann.get("fail_annotation", "0")
        item["fail_category"] = ann.get("fail_category", "")
        item["fail_reason"] = ann.get("fail_reason", "")
        item["ideal_action"] = ann.get("ideal_action", "")
    return history


def transform_from_txt(raw_url: str, annotations: list[dict]) -> dict:
    """Build output dict from a MAST console_log.txt source."""
    base_url = raw_url.rsplit("/", 1)[0]

    console_log_text = fetch_text(raw_url)

    try:
        question = fetch_text(f"{base_url}/prompt.txt").strip()
    except Exception:
        question = ""

    try:
        ground_truth = fetch_text(f"{base_url}/expected_answer.txt").strip()
    except Exception:
        ground_truth = ""

    turns = parse_console_log(console_log_text)

    history = [
        {
            "content": t["content"],
            "role": t["role"],
            "step": str(idx),
        }
        for idx, t in enumerate(turns)
    ]
    history = _apply_annotations(history, annotations)

    return {
        "is_correct": None,
        "question": question,
        "ground_truth": ground_truth,
        "history": history,
        "system_prompt": {},
    }


def transform_from_json(raw_data: dict, annotations: list[dict]) -> dict:
    """Build output dict from a JSON-based (Algorithm-Generated) source."""
    history = [
        {
            "content": item.get("content", ""),
            "role": item.get("name") or item.get("role", ""),
            "step": str(idx),
        }
        for idx, item in enumerate(raw_data.get("history", []))
    ]
    history = _apply_annotations(history, annotations)

    return {
        "is_correct": raw_data.get("is_correct"),
        "question": raw_data.get("question", ""),
        "ground_truth": raw_data.get("ground_truth", ""),
        "history": history,
        "system_prompt": raw_data.get("system_prompt", {}),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build annotated JSON from automatic or manual annotation source."
    )
    parser.add_argument("--input_json", required=True, help="Source JSON (e.g. MP-Bench/1/automatic/1.json)")
    parser.add_argument("--output", required=True, help="Output path for the transformed JSON")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    with input_path.open("r", encoding="utf-8") as f:
        source = json.load(f)

    log_source: str = source["log_source"]
    raw_url = github_tree_to_raw(log_source)
    annotations: list[dict] = source.get("annotation", [])

    # Generalized detection:
    # - Try JSON first (works for automatic + manual entries pointing to JSON logs)
    # - If parsing fails, treat as console text log
    raw_text = fetch_text(raw_url)
    try:
        raw_data = json.loads(raw_text)
        transformed = transform_from_json(raw_data, annotations)
    except json.JSONDecodeError:
        transformed = transform_from_txt(raw_url, annotations)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
