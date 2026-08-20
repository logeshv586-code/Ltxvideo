# Contributing to Ltxvideo

Thanks for helping improve Ltxvideo. The project focuses on practical local AI video generation, especially constrained laptop-class NVIDIA hardware.

## Good places to start

New contributors can begin with the issues labeled [`good first issue`](https://github.com/logeshv586-code/Ltxvideo/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22).

Current beginner-friendly tasks include:

- Windows RTX 4050 setup checklist: #4
- Copy-ready example prompts for every Studio: #5
- Lightweight startup diagnostics command: #6

The first real RTX 4050 demo reel is tracked in #7.

## Development setup

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the lightweight tests

The normal unit tests do not require loading the full video model:

```bash
python -m unittest discover -s tests -v
```

Syntax check:

```bash
python -m py_compile app.py config.py run.py download_models.py engine/*.py tests/*.py
```

## Contribution guidelines

1. Keep changes focused and easy to review.
2. Preserve the low-memory RTX 4050 / 16 GB RAM design target unless a change is explicitly introducing another hardware profile.
3. Do not describe FFmpeg delivery resizing as native 720p/1080p diffusion.
4. Do not claim a generation result was produced on specific hardware unless it was actually rendered there.
5. Add or update tests when changing prompt logic, frame/resolution constraints, memory behavior, or video QC.
6. Avoid committing downloaded model weights, caches, generated videos, secrets, or machine-specific files.
7. Clearly identify third-party code or assets and preserve their licensing requirements.

## Pull requests

A useful pull request should include:

- what changed
- why it improves the project
- how it was tested
- any GPU/model validation that was not possible
- screenshots or short clips when the UI or output changes

Small documentation and test improvements are welcome.

## Licensing

By contributing code to this repository, you agree that your contribution may be distributed under the repository's Apache-2.0 license. LTX model weights and third-party components remain subject to their own upstream licenses and terms.
