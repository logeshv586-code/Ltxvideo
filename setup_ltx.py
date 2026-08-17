"""Install the official LTX-2 runtime used by this app.

The repository is pinned to a known compatible upstream commit so API changes on
LTX main do not randomly break this project.
"""

from pathlib import Path
import subprocess
import sys

from config import LTX_SOURCE_DIR

UPSTREAM = "https://github.com/Lightricks/LTX-2.git"
PINNED_COMMIT = "400fd31054597515f47125691032c04b1c3ee24e"


def run(*args: str, cwd: Path | None = None) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    LTX_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not (LTX_SOURCE_DIR / ".git").exists():
        if LTX_SOURCE_DIR.exists() and any(LTX_SOURCE_DIR.iterdir()):
            raise RuntimeError(
                f"{LTX_SOURCE_DIR} exists but is not a Git checkout. Remove or rename it, then rerun setup_ltx.py."
            )
        run("git", "clone", UPSTREAM, str(LTX_SOURCE_DIR))

    run("git", "fetch", "origin", PINNED_COMMIT, cwd=LTX_SOURCE_DIR)
    run("git", "checkout", "--detach", PINNED_COMMIT, cwd=LTX_SOURCE_DIR)

    # Install official packages. Their pyproject files define the compatible
    # Torch/Transformers stack.
    run(sys.executable, "-m", "pip", "install", "-e", str(LTX_SOURCE_DIR / "packages" / "ltx-core"))
    run(sys.executable, "-m", "pip", "install", "-e", str(LTX_SOURCE_DIR / "packages" / "ltx-pipelines"))

    print("\n✅ Official LTX-2 runtime installed.")
    print(f"Pinned upstream commit: {PINNED_COMMIT}")
    print("Next: python download_models.py")


if __name__ == "__main__":
    main()
