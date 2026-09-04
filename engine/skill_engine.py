"""Reusable skill engine applied to every LTX generation request.

The engine keeps the user's literal request as the source of truth, then adds only
supporting direction that is missing: reference anchoring, temporal budgeting,
camera stability, mode-specific consistency, hardware warnings, negative-prompt
hardening, and prompt quality gates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# The bundled T5 encoder is limited to 128 tokens. Words are not tokens, so
# keep a material margin for punctuation and character names.
MAX_LTX_PROMPT_WORDS = 80

CAMERA_WORDS = (
    "camera", "dolly", "pan ", "panning", "tilt", "orbit", "zoom", "handheld",
    "tracking", "track ", "crane", "jib", "static", "locked-off", "locked off",
    "push-in", "push in", "pull-out", "pull out", "whip pan", "fpv", "overhead",
)

MODE_NEGATIVES = {
    "general": "unintended camera movement, incoherent scene changes",
    "comics": "style drift, line-art drift, changing costume, changing face, inconsistent palette",
    "real_world": "rubbery motion, floating objects, plastic skin, warped anatomy, impossible physics",
    "action": "unreadable choreography, broken contact, teleporting limbs, impossible trajectories, chaotic camera",
    "cartoon": "character drift, changing costume, changing colors, inconsistent proportions, style drift",
}

MODE_SUPPORT = {
    "comics": (
        "Keep line work, silhouette, facial design, costume, props, and the color palette visually consistent throughout the shot."
    ),
    "real_world": (
        "Preserve believable anatomy, material response, weight, inertia, contact, and coherent background geometry."
    ),
    "action": (
        "Prioritize readable choreography: preserve body mechanics, contact points, momentum, object trajectories, and screen direction."
    ),
    "cartoon": (
        "Preserve the same character identity, face, clothing, colors, proportions, props, environment treatment, and visual style."
    ),
}


@dataclass(frozen=True)
class VideoRequest:
    raw_prompt: str
    mode: str = "auto"
    duration_seconds: float = 4.0
    width: int = 384
    height: int = 224
    num_frames: int = 121
    has_reference: bool = False
    negative_prompt: str = ""
    character_lock: str = ""
    reference_role: str = "identity/composition"


@dataclass
class SkillPlan:
    prompt: str
    negative_prompt: str
    applied_skills: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    locked_terms: list[str] = field(default_factory=list)
    quality_score: int = 100

    def trace_text(self) -> str:
        applied = " → ".join(self.applied_skills) if self.applied_skills else "none"
        lines = [f"Skills: {applied}", f"Prompt quality gate: {self.quality_score}/100"]
        if self.locked_terms:
            lines.append("Intent locks: " + ", ".join(self.locked_terms[:8]))
        if self.warnings:
            lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


class VideoSkillEngine:
    """Deterministic skill router. No cloud LLM or external API is required."""

    def plan(self, request: VideoRequest) -> SkillPlan:
        raw = " ".join((request.raw_prompt or "").strip().split())
        if not raw:
            raise ValueError("A video prompt is required.")

        mode = (request.mode or "auto").strip().lower()
        if mode == "auto":
            mode = self._detect_mode(raw)
        if mode not in {"general", "comics", "real_world", "action", "cartoon"}:
            mode = "general"

        fragments: list[str] = [raw]
        applied = ["intent-lock"]
        warnings: list[str] = []
        locked_terms = self._extract_locks(raw)

        if request.has_reference:
            fragments.append(
                "Treat the supplied reference image as authoritative for the opening composition and "
                f"{request.reference_role}; preserve those details except where the user's request explicitly asks for a change."
            )
            applied.append("reference-anchor")

        if request.character_lock.strip():
            fragments.append(
                "Continuity lock: " + " ".join(request.character_lock.strip().split()) + "."
            )
            applied.append("continuity-lock")

        temporal, temporal_warning = self._temporal_direction(raw, request.duration_seconds, mode)
        if temporal:
            fragments.append(temporal)
            applied.append("temporal-budget")
        if temporal_warning:
            warnings.append(temporal_warning)

        if not self._has_camera_direction(raw):
            fragments.append(
                "Keep the camera stable and motivated; do not introduce an unrequested camera move."
            )
            applied.append("camera-stability")
        else:
            applied.append("camera-lock")

        support = MODE_SUPPORT.get(mode)
        if support and support.lower() not in raw.lower():
            fragments.append(support)
            applied.append(f"{mode}-consistency")

        hardware_warning = self._hardware_warning(request)
        if hardware_warning:
            warnings.append(hardware_warning)
        applied.append("rtx4050-budget")

        prompt = self._compile_prompt(raw, fragments[1:], warnings)
        negative = self._merge_negative(request.negative_prompt, MODE_NEGATIVES.get(mode, ""))
        applied.append("negative-guard")

        score, quality_warnings = self._quality_gate(prompt, raw, request.duration_seconds)
        warnings.extend(quality_warnings)
        applied.append("prompt-quality-gate")

        return SkillPlan(
            prompt=prompt,
            negative_prompt=negative,
            applied_skills=applied,
            warnings=self._dedupe(warnings),
            locked_terms=locked_terms,
            quality_score=score,
        )

    @staticmethod
    def _detect_mode(raw: str) -> str:
        lower = raw.lower()
        if "cartoon story scene" in lower or "character bible:" in lower:
            return "cartoon"
        if "action environment:" in lower or "energy:" in lower or "readable choreography" in lower:
            return "action"
        if "realism profile:" in lower or "realistic anatomy" in lower or "believable weight" in lower:
            return "real_world"
        if "visual direction:" in lower and ("comic" in lower or "manga" in lower or "line work" in lower):
            return "comics"
        return "general"

    @staticmethod
    def _extract_locks(raw: str) -> list[str]:
        quoted = re.findall(r'["“”\']([^"“”\']+)["“”\']', raw)
        numbers = re.findall(r"\b\d+(?:\.\d+)?(?:mm|fps|k|p|x|%|s|sec|seconds?)?\b", raw, flags=re.I)
        colors = re.findall(
            r"\b(?:red|blue|green|yellow|orange|purple|violet|pink|black|white|gray|grey|gold|silver|teal|cyan|magenta|brown)\b",
            raw,
            flags=re.I,
        )
        return VideoSkillEngine._dedupe([*quoted, *numbers, *colors])

    @staticmethod
    def _has_camera_direction(raw: str) -> bool:
        lower = f" {raw.lower()} "
        return any(word in lower for word in CAMERA_WORDS)

    @staticmethod
    def _temporal_direction(raw: str, duration: float, mode: str) -> tuple[str, str | None]:
        connectors = len(re.findall(r"\b(?:then|after|before|while|followed by|next|finally|suddenly)\b", raw, flags=re.I))
        if duration <= 4.5:
            text = "Fit the requested motion into one clear primary action beat with an immediately readable start and end state."
            limit = 1
        elif duration <= 6.5:
            text = "Keep the shot to one primary action beat plus one natural follow-through; avoid unrelated secondary events."
            limit = 2
        else:
            text = "Stage the requested events chronologically with no more than two major action beats and one natural follow-through."
            limit = 3
        warning = None
        if connectors > limit:
            warning = (
                f"The prompt appears to contain {connectors + 1} temporal beats for a {duration:.1f}s clip; "
                "consider splitting it into multiple shots for stronger adherence."
            )
        if mode == "action" and duration <= 6.5:
            text += " Use only one major camera movement during the action."
        return text, warning

    @staticmethod
    def _hardware_warning(request: VideoRequest) -> str | None:
        pixels = request.width * request.height
        if request.num_frames >= 241:
            return "241 frames is the heaviest exposed native clip size for the RTX 4050 profile."
        if request.num_frames >= 193 and pixels >= 512 * 288:
            return "Long duration plus 512×288-class resolution may pressure 6 GB VRAM; close other GPU apps before generating."
        return None

    @staticmethod
    def _compile_prompt(raw: str, additions: Iterable[str], warnings: list[str]) -> str:
        raw_words = raw.split()
        if len(raw_words) >= MAX_LTX_PROMPT_WORDS:
            warnings.append(
                "The prompt was shortened to fit LTX's 128-token text-encoder limit."
            )
            return " ".join(raw_words[:MAX_LTX_PROMPT_WORDS])

        result = raw
        for addition in additions:
            candidate = f"{result} {addition}".strip()
            if len(candidate.split()) <= MAX_LTX_PROMPT_WORDS:
                result = candidate
            else:
                warnings.append("Some optional skill direction was omitted to keep the compiled prompt within 200 words.")
                break
        return result

    @staticmethod
    def _merge_negative(user_negative: str, mode_negative: str) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for block in (user_negative, mode_negative):
            for part in (block or "").split(","):
                cleaned = " ".join(part.strip().split())
                key = cleaned.lower()
                if cleaned and key not in seen:
                    seen.add(key)
                    parts.append(cleaned)
        return ", ".join(parts)

    @staticmethod
    def _quality_gate(prompt: str, raw: str, duration: float) -> tuple[int, list[str]]:
        score = 100
        warnings: list[str] = []
        lower = prompt.lower()
        camera_moves = sum(lower.count(term) for term in ("dolly", "pan ", "zoom", "orbit", "crane", "handheld", "tracking"))
        if "static" in lower and camera_moves:
            score -= 15
            warnings.append("Prompt mixes a static-camera instruction with camera movement; remove one if results drift.")
        if camera_moves > 2 and duration <= 6.5:
            score -= 15
            warnings.append("More than two camera-motion cues were detected in a short clip.")
        if len(prompt.split()) > MAX_LTX_PROMPT_WORDS:
            score -= 25
            warnings.append("Compiled prompt exceeds the recommended 200-word LTX range.")
        if raw not in prompt:
            score -= 40
            warnings.append("Intent-lock failure: the original user prompt was not preserved verbatim.")
        return max(0, score), warnings

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(value.strip())
        return out


SKILL_ENGINE = VideoSkillEngine()
