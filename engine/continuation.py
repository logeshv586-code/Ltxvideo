"""Backward-compatible import for the new Cartoon Story continuity engine."""
from engine.storyboard import CartoonStoryGenerator, build_scene_prompt, split_story_beats

ContinuationGenerator = CartoonStoryGenerator

__all__ = ["ContinuationGenerator", "CartoonStoryGenerator", "build_scene_prompt", "split_story_beats"]
