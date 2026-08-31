"""Verbosity steering: the one lever allowed to touch the system prompt.

Steering text is appended once per conversation and replayed byte-for-byte on
every later turn (see :class:`~headroom_output.contract.PromptDecision`). That
is why this is a :class:`PromptLever` rather than a :class:`ParamLever`: a
lever that could see the turn kind would be tempted to vary the text turn to
turn, which is exactly the cache-busting mistake the contract makes
unrepresentable for param levers. Steering intentionally gives up that
per-turn precision to keep the system prompt — and the provider prefix cache
built on it — stable for the life of the conversation.

The level text below is copied byte-for-byte from
``headroom/proxy/output_verbosity_policy.py``. It must stay byte-stable across
releases: it lands in the system prompt, so any edit invalidates the prefix
cache for every conversation currently running that text, at roughly 60x the
cost of whatever tokens the rewording might have saved. Do not reword it here.
"""

from __future__ import annotations

from ..contract import ConversationFeatures, LeverDescriptor, PromptDecision, WireFormat

# Levels are cumulative: each includes everything above it. Verbatim from
# headroom/proxy/output_verbosity_policy.py — do not reword.
VERBOSITY_LEVELS: dict[int, str] = {
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
        "prose to the minimum needed to be unambiguous."
    ),
    4: (
        "Minimum tokens. Fragments fine. No preamble, no postamble, no "
        "restating context, no rationale. Answer, smallest-possible edits, "
        "nothing else."
    ),
}


class VerbosityLever:
    """Appends a fixed verbosity instruction to the system-prompt tail.

    One instance per level; level 0 is "decline" rather than a fifth string,
    so a caller that wants no steering simply does not register this lever
    (or registers it with ``level=0``, which always returns an empty
    decision — kept as a valid no-op rather than a constructor error, so a
    config value of 0 does not need special-casing by the wiring code).
    """

    def __init__(self, level: int) -> None:
        if not 0 <= level <= 4:
            raise ValueError(f"verbosity level must be 0-4, got {level}")
        self.level = level
        self.descriptor = LeverDescriptor(
            # Stable across levels on purpose: the descriptor name keys attribution
            # history, so baking the level into it would orphan every prior
            # observation the moment the level changed. The level rides the
            # decision label instead, where a change is a new data point
            # rather than a new lever.
            name="verbosity",
            summary=f"Append level-{level} verbosity steering to the system prompt tail.",
            wire_formats=(
                WireFormat.ANTHROPIC,
                WireFormat.OPENAI_CHAT,
                WireFormat.OPENAI_RESPONSES,
            ),
            mutates_cache_key=True,
        )

    def decide_conversation(self, features: ConversationFeatures) -> PromptDecision:
        """Return this level's steering text, or decline at level 0.

        ``features`` is unused beyond satisfying the protocol: the text is a
        pure function of ``self.level`` by design — reacting to the model or
        harness here would make the "byte-stable per conversation" contract a
        matter of luck rather than a guarantee.
        """
        if self.level == 0:
            return PromptDecision()
        return PromptDecision(
            system_suffix=VERBOSITY_LEVELS[self.level],
            label=f"verbosity:L{self.level}",
        )
