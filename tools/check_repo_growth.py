"""Check public GitHub discoverability settings for Ltxvideo."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

REPO_API = "https://api.github.com/repos/logeshv586-code/Ltxvideo"
REQUIRED_TOPICS = {
    "ltx-video",
    "ai-video",
    "video-generation",
    "image-to-video",
    "text-to-video",
    "rtx-4050",
    "gradio",
    "diffusers",
    "generative-ai",
    "cartoon-animation",
}


def fetch_repo() -> dict:
    request = urllib.request.Request(
        REPO_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Ltxvideo-growth-health",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def evaluate(repo: dict) -> list[tuple[str, bool, str]]:
    description = (repo.get("description") or "").strip()
    topics = set(repo.get("topics") or [])
    discussions = bool(repo.get("has_discussions"))
    missing_topics = sorted(REQUIRED_TOPICS - topics)
    return [
        ("Description", bool(description), description or "missing"),
        ("Topics", not missing_topics, "complete" if not missing_topics else f"missing: {', '.join(missing_topics)}"),
        ("Discussions", discussions, "enabled" if discussions else "disabled"),
    ]


def main() -> int:
    try:
        repo = fetch_repo()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[FAIL] Could not query GitHub: {exc}")
        return 2

    checks = evaluate(repo)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(
        f"[INFO] stars={repo.get('stargazers_count', 0)} "
        f"forks={repo.get('forks_count', 0)} "
        f"watchers={repo.get('subscribers_count', 0)}"
    )
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
