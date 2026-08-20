## What changed

<!-- Explain the user-visible or developer-visible change. -->

## Why

<!-- Link an issue/discussion when possible and explain the problem being solved. -->

## Validation

- [ ] `python -m unittest discover -s tests -v`
- [ ] `python -m py_compile app.py config.py run.py download_models.py engine/*.py tests/*.py`
- [ ] I did not add model weights, generated outputs, secrets, caches, or local environment files.
- [ ] I kept claims about RTX 4050 support and native vs delivery resolution accurate.

## GPU validation

<!-- If this changes generation/runtime behavior, include GPU, VRAM/RAM, resolution, frames, steps, seed, and observed result. If GPU validation is not needed, say why. -->

## Screenshots / demo

<!-- For UI changes or verified generation improvements, add concise evidence when available. Do not label cloud-generated media as local RTX 4050 output. -->

## Contributor checklist

- [ ] The change is focused and does not include unrelated files.
- [ ] Documentation is updated when behavior changes.
- [ ] New behavior has tests where practical.
- [ ] Third-party code/assets and licenses are respected.
