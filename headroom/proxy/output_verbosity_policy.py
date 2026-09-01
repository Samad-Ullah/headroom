"""Pure output verbosity steering policy."""

from __future__ import annotations

# Sentinel prefix marks the steering block so application is idempotent and
# the block is recognizable in logs/diffs.
STEERING_SENTINEL = "<headroom_output_shaping>"
STEERING_SUFFIX = "</headroom_output_shaping>"

# Levels are cumulative: each includes everything above it. Text must stay
# byte-stable across releases for prefix-cache friendliness; edits to these
# strings are cache-busting changes.
#
# L3 and L4 carry two rules L1/L2 do not, because only they instruct the model
# to drop content rather than ceremony:
#
#   * A verbatim list. "Omit rationale" and "fragments fine" invite a model to
#     paraphrase an error string or round a number. The dangerous case is a
#     dropped negation -- losing "not" from "does not retry" inverts the
#     answer while making it shorter, which is exactly what the instruction
#     rewards.
#   * A clarity exception. Without it these levels are tersest precisely where
#     terseness is most costly: destructive actions, security warnings, and
#     multi-step sequences.
#
# L1/L2 only forbid preamble and restating context, so they carry neither --
# every token in this block is paid on every request of every conversation.
VERBOSITY_LEVELS = {
    1: (
        "Skip preamble and postamble. Do not announce what you are about to "
        "do or recap what you just did; start with the substance."
    ),
    2: (
        "Skip preamble and postamble; start with the substance. Never restate "
        "code, file contents, diffs, or tool output that already appear in "
        "this conversation — reference them by path and line instead. After a "
        "tool call succeeds, continue without narrating the result."
    ),
    3: (
        "Skip preamble and postamble. Never restate code, file contents, "
        "diffs, or tool output already in this conversation — reference by "
        "path and line. Give conclusions only; omit rationale unless the user "
        "asks why. Prefer the smallest edit over rewriting whole files. Keep "
        "prose to the minimum needed to be unambiguous. Reproduce code, error "
        "strings, numbers, units, and API or CLI names exactly, and never drop "
        "a negation (not, never, no, only, except) to save words. Use full "
        "prose for destructive or irreversible actions, security warnings, and "
        "any multi-step sequence where brevity would create ambiguity."
    ),
    4: (
        "Minimum tokens. Fragments fine. No preamble, no postamble, no "
        "restating context, no rationale. Answer, smallest-possible edits, "
        "nothing else. Reproduce code, error strings, numbers, units, and API "
        "or CLI names exactly, and never drop a negation (not, never, no, "
        "only, except). Use full prose for destructive or irreversible "
        "actions, security warnings, and any multi-step sequence where "
        "brevity would create ambiguity."
    ),
}


def steering_text(level: int) -> str | None:
    """The full steering block for a verbosity level, or ``None`` for level 0."""
    text = VERBOSITY_LEVELS.get(level)
    if text is None:
        return None
    return f"{STEERING_SENTINEL}\n{text}\n{STEERING_SUFFIX}"


def replace_or_append_steering_block(existing: str, block: str) -> tuple[str, bool]:
    """Replace an existing steering block in text, or append one at the tail."""
    start = existing.find(STEERING_SENTINEL)
    if start >= 0:
        end = existing.find(STEERING_SUFFIX, start)
        end = len(existing) if end < 0 else end + len(STEERING_SUFFIX)
        prefix = existing[:start].rstrip()
        suffix = existing[end:].lstrip("\n")
        parts = [part for part in (prefix, block, suffix) if part]
        updated = "\n\n".join(parts)
        return updated, updated != existing

    updated = f"{existing.rstrip()}\n\n{block}" if existing.strip() else block
    return updated, updated != existing
