# Video Skill Engine

The Video Skill Engine is the mandatory directing layer between every UI generation request and LTX inference.

It is intentionally **local and deterministic**: it does not require a cloud LLM, API key, or a second large language model in RAM. That matters for the RTX 4050 / 16 GB RAM target.

## Why this architecture exists

Modern AI-video studios expose more than a single prompt box. Their public workflows separate intent into reusable controls such as visual references, start/end anchors, camera movement, lens/framing, duration, style, motion, and continuity. The important engineering lesson is not to copy any proprietary backend; it is to stop making the video model guess information that can be locked before inference.

Public references used when designing this layer:

- Higgsfield Camera Controls and Cinema Studio: explicit camera/motion controls, visual references, start/end frames, and reusable cinematic settings.
- Higgsfield Popcorn storyboard flow: multiple numbered visual references and continuity across frames.
- Runway Image-to-Video prompting: the input image establishes composition/style/subject; the text prompt should focus on what changes, motion, camera work, and temporal progression.
- Adobe Firefly video prompting: shot type + character + action + location + aesthetic, with explicit camera and temporal language.
- Lightricks LTX-Video: detailed chronological prompts, literal actions, camera movement, environment and lighting, kept within roughly 200 words.

## Always-on skill stack

Every call to `VideoGenerator.generate_text_to_video()` and `generate_image_to_video()` now passes through these skills before the diffusion pipeline:

1. **Intent Lock** — keeps the user's compiled request verbatim at the start of the final prompt; also records quoted text, numbers and explicit colors as high-value locks.
2. **Reference Anchor** — on I2V, treats the uploaded image as authoritative for the opening composition and identity unless the prompt explicitly requests a change.
3. **Continuity Lock** — optional persistent identity/style details can be injected without replacing the original request.
4. **Temporal Budget** — scales action complexity to the actual clip length. Short clips get one primary action beat rather than a dense sequence that the model cannot stage coherently.
5. **Camera Lock / Stability** — explicit user camera directions are preserved; when none is supplied the system prevents random unrequested camera moves.
6. **Mode Consistency** — automatically recognizes Comics, Real World, Action, Cartoon Story, or General prompts and applies only the relevant consistency rules.
7. **RTX 4050 Budget** — warns when duration + resolution is near the heavy end of the laptop profile without silently changing the user's settings.
8. **Negative Guard** — combines the user's negative prompt with mode-specific failure guards while removing duplicates.
9. **Prompt Quality Gate** — checks the 200-word LTX guidance, conflicting camera cues, overloaded short shots, and verifies the original request still exists in the compiled prompt.
10. **Technical Video QC** — after export, verifies the MP4 can be decoded, checks dimensions/duration, samples for black frames and flags near-frozen motion.

## Mode auto-routing

The engine can be called with `mode="auto"`. It recognizes the prompt grammar already produced by the UI's mode-specific prompt directors:

- `Cartoon story scene ...` / `Character bible:` → Cartoon continuity skill
- `Action environment:` / `Energy:` → Action choreography skill
- `Realism profile:` / realistic-physics anchors → Real-world skill
- `Visual direction:` + comic/manga/line-work anchors → Comics skill
- otherwise → General

This means the skill engine is active even if a new UI route is added later and it still calls the shared generator.

## Generation flow

```text
User prompt + structured UI fields + optional reference image
                           │
                           ▼
                    INTENT LOCK
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      REFERENCE SKILL               MODE ROUTER
             │                 comics / real / action /
             │                    cartoon / general
             └─────────────┬─────────────┘
                           ▼
                  TEMPORAL BUDGET
                           ▼
              CAMERA / CONTINUITY LOCK
                           ▼
                   NEGATIVE GUARD
                           ▼
                  PROMPT QUALITY GATE
                           ▼
                RTX 4050 MEMORY BUDGET
                           ▼
                     LTX INFERENCE
                           ▼
                 TECHNICAL VIDEO QC
                           ▼
                     MP4 / EXPORT
```

## What the engine does **not** claim

Technical QC can verify a playable file, duration, resolution, black frames and obvious freezing. It cannot prove that a generated actor's expression or every object exactly matches the text. A semantic evaluator would require a separate vision-language model and additional memory. On the target 16 GB laptop that should be optional rather than silently consuming RAM.

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py config.py run.py download_models.py engine/*.py
```

Key tests cover intent preservation, I2V reference anchoring, action timing, camera locking, automatic mode recognition, negative-prompt deduplication, prompt-length gating, and MP4 frozen/decodability checks.
