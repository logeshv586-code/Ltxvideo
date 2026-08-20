# Security Policy

## Supported code

Security fixes are accepted against the current `main` branch. This repository is an application layer around upstream model/runtime dependencies, so vulnerabilities in LTX-Video, PyTorch, Diffusers, Transformers, Gradio, FFmpeg, Hugging Face tooling, or other third-party packages may also need to be reported upstream.

## Reporting a vulnerability

Please do **not** publish credentials, access tokens, private model URLs, personal data, or an exploitable proof-of-concept in a public issue or Discussion.

If GitHub private vulnerability reporting is enabled for this repository, use **Security → Report a vulnerability**. Otherwise, open a minimal public issue that only states that a security concern exists and which project area it affects, without exploit details or secrets, so the maintainer can coordinate a private channel.

## Scope

Useful reports include issues involving unsafe file handling, command execution, dependency installation behavior, path traversal, unintended network exposure, secret leakage, or insecure defaults in this repository's own application code.

Model quality problems, prompt adherence, CUDA out-of-memory errors, and ordinary generation failures are not security vulnerabilities; use Issues or Discussions for those.
