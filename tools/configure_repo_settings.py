"""Configure Ltxvideo repository discovery settings with an admin-capable GitHub token.

Usage (PowerShell):
  $env:GITHUB_TOKEN = "<fine-grained token with Administration: write>"
  python tools/configure_repo_settings.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

REPO_API = "https://api.github.com/repos/logeshv586-code/Ltxvideo"
TOPICS_API = REPO_API + "/topics"
DESCRIPTION = (
    "Local LTX-Video AI studio for RTX 4050 laptops — text/image-to-video, "
    "Action, Comics, Real-World and Cartoon Story continuity."
)
TOPICS = [
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
]


def _request(url: str, token: str, method: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Ltxvideo-repo-configurator",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def configure(token: str) -> tuple[dict, dict]:
    repo = _request(
        REPO_API,
        token,
        "PATCH",
        {"description": DESCRIPTION, "has_discussions": True},
    )
    topics = _request(TOPICS_API, token, "PUT", {"names": TOPICS})
    return repo, topics


def main() -> int:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        print(
            "[FAIL] Set GITHUB_TOKEN or GH_TOKEN to a token with repository Administration: write permission.",
            file=sys.stderr,
        )
        return 2
    try:
        repo, topics = configure(token)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[FAIL] GitHub returned HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"[FAIL] Could not reach GitHub: {exc}", file=sys.stderr)
        return 1

    print(f"[PASS] Description: {repo.get('description')}")
    print(f"[PASS] Discussions: {'enabled' if repo.get('has_discussions') else 'disabled'}")
    print(f"[PASS] Topics: {', '.join(topics.get('names') or [])}")
    print("Run: python tools/check_repo_growth.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
