"""Benchmark task generator: small Python packages with a seeded bug.

Each task is a self-contained package plus a pytest suite in which the large
majority of tests pass and one or two fail. The agent must run the suite, read
the failure, locate the bug and fix it.

The suite is deliberately large (``n_pass`` filler tests) because the object of
study is what happens to a *bulky test log* on its way to the model. A task
whose test output is three lines would not exercise the compression path at all.

Difficulty is held roughly constant: every bug is a single-token edit inside one
function, reachable from the assertion text in the failure.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Bug:
    key: str
    # The correct line, and the corrupted line that replaces it.
    correct: str
    broken: str
    description: str


# Each bug is a single-token mutation of one line: an operator flip, an
# off-by-one, an inverted guard, a wrong constant. All are diagnosable from the
# failing assertion without reading the whole package.
BUGS: list[Bug] = [
    Bug("off_by_one", "return items[:n]", "return items[: n - 1]",
        "slice drops the last element"),
    Bug("operator_flip", "return total * rate", "return total / rate",
        "multiplication became division"),
    Bug("inverted_guard", "if value is None:", "if value is not None:",
        "None-guard inverted"),
    Bug("wrong_rounding", "return round(value, 2)", "return round(value, 1)",
        "rounds to the wrong precision"),
    Bug("bad_default", "def scale(x, factor=2):", "def scale(x, factor=3):",
        "wrong default argument"),
    Bug("off_by_one_range", "for i in range(len(rows)):", "for i in range(len(rows) - 1):",
        "loop skips the final row"),
    Bug("comparison_flip", "if count >= limit:", "if count > limit:",
        "boundary comparison excludes the limit"),
    Bug("wrong_accumulator", "total += item", "total -= item",
        "accumulator subtracts"),
    Bug("string_case", "return name.strip().lower()", "return name.strip().upper()",
        "normalises to the wrong case"),
    Bug("missing_abs", "return abs(a - b)", "return a - b",
        "distance can go negative"),
    Bug("percent_scale", "return part / whole * 100", "return part / whole * 10",
        "percentage scaled by the wrong factor"),
    Bug("empty_default", "return sorted(set(tags))", "return sorted(tags)",
        "duplicates are not removed"),
    Bug("index_start", "return text[1:]", "return text[2:]",
        "strips one character too many"),
    Bug("min_max_swap", "return max(lo, min(value, hi))", "return min(lo, max(value, hi))",
        "clamp bounds transposed"),
    Bug("truthy_zero", "if value or value == 0:", "if value:",
        "zero treated as missing"),
    Bug("join_sep", 'return ", ".join(parts)', 'return ",".join(parts)',
        "separator missing its space"),
    Bug("keys_vs_values", "return list(mapping.values())", "return list(mapping.keys())",
        "returns keys instead of values"),
    Bug("float_div", "return count // size", "return count / size",
        "integer division became float division"),
    Bug("reverse_sort", "return sorted(scores, reverse=True)", "return sorted(scores)",
        "sort order not reversed"),
    Bug("strip_chars", "return path.rstrip('/')", "return path.strip('/')",
        "strips the leading slash too"),
]

MODULE = '''\
"""Small utility library used by the reporting pipeline."""


def take_first(items, n):
    """Return the first ``n`` items."""
    return items[:n]


def apply_rate(total, rate):
    """Scale ``total`` by ``rate``."""
    return total * rate


def coalesce(value, fallback):
    """Return ``fallback`` when ``value`` is missing."""
    if value is None:
        return fallback
    return value


def to_money(value):
    """Round a monetary amount to cents."""
    return round(value, 2)


def scale(x, factor=2):
    """Multiply ``x`` by ``factor``."""
    return x * factor


def row_ids(rows):
    """Collect the id of every row."""
    out = []
    for i in range(len(rows)):
        out.append(rows[i]["id"])
    return out


def over_limit(count, limit):
    """True when ``count`` has reached ``limit``."""
    if count >= limit:
        return True
    return False


def total_of(items):
    """Sum ``items``."""
    total = 0
    for item in items:
        total += item
    return total


def normalise(name):
    """Canonical form of a display name."""
    return name.strip().lower()


def distance(a, b):
    """Absolute distance between two numbers."""
    return abs(a - b)


def percent(part, whole):
    """Express ``part`` as a percentage of ``whole``."""
    return part / whole * 100


def unique_tags(tags):
    """Sorted, de-duplicated tags."""
    return sorted(set(tags))


def drop_prefix(text):
    """Drop the single leading marker character."""
    return text[1:]


def clamp(value, lo, hi):
    """Constrain ``value`` to [lo, hi]."""
    return max(lo, min(value, hi))


def present(value, fallback):
    """Return ``value`` when supplied, treating 0 as supplied."""
    if value or value == 0:
        return value
    return fallback


def render_list(parts):
    """Human-readable comma-separated list."""
    return ", ".join(parts)


def config_values(mapping):
    """The configured values."""
    return list(mapping.values())


def pages_needed(count, size):
    """Whole pages required to hold ``count`` rows."""
    return count // size


def leaderboard(scores):
    """Scores, highest first."""
    return sorted(scores, reverse=True)


def normalise_path(path):
    """Drop any trailing slash, keeping the leading one."""
    return path.rstrip('/')
'''

# One real assertion per function; these are the tests that expose the bugs.
REAL_TESTS = '''\
from reportlib import (
    take_first, apply_rate, coalesce, to_money, scale,
    row_ids, over_limit, total_of, normalise, distance,
    percent, unique_tags, drop_prefix, clamp, present,
    render_list, config_values, pages_needed, leaderboard, normalise_path,
)


def test_take_first_returns_n_items():
    assert take_first([1, 2, 3, 4, 5], 3) == [1, 2, 3]


def test_apply_rate_scales_up():
    assert apply_rate(100.0, 1.5) == 150.0


def test_coalesce_replaces_missing():
    assert coalesce(None, "fallback") == "fallback"
    assert coalesce("value", "fallback") == "value"


def test_to_money_rounds_to_cents():
    assert to_money(48.387096774193544) == 48.39


def test_scale_default_factor_is_two():
    assert scale(21) == 42


def test_row_ids_includes_every_row():
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert row_ids(rows) == [1, 2, 3]


def test_over_limit_is_inclusive():
    assert over_limit(10, 10) is True


def test_total_of_sums():
    assert total_of([1, 2, 3, 4]) == 10


def test_normalise_lowercases():
    assert normalise("  Ada Lovelace  ") == "ada lovelace"


def test_distance_is_absolute():
    assert distance(3, 10) == 7


def test_percent_of_whole():
    assert percent(25, 200) == 12.5


def test_unique_tags_deduplicates():
    assert unique_tags(["b", "a", "b"]) == ["a", "b"]


def test_drop_prefix_removes_one_char():
    assert drop_prefix("#tag") == "tag"


def test_clamp_constrains_to_range():
    assert clamp(15, 0, 10) == 10
    assert clamp(-5, 0, 10) == 0


def test_present_treats_zero_as_supplied():
    assert present(0, "fallback") == 0


def test_render_list_uses_comma_space():
    assert render_list(["a", "b"]) == "a, b"


def test_config_values_returns_values():
    assert config_values({"a": 1, "b": 2}) == [1, 2]


def test_pages_needed_is_integer():
    assert pages_needed(10, 3) == 3


def test_leaderboard_is_descending():
    assert leaderboard([1, 5, 3]) == [5, 3, 1]


def test_normalise_path_keeps_leading_slash():
    assert normalise_path("/a/b/") == "/a/b"
'''


def _filler_tests(n: int) -> str:
    """Passing tests, so the failure sits in a realistically bulky log."""
    body = [
        "\n",
        "# Regression cover for the reporting pipeline. These all pass; they exist\n",
        "# so the suite produces a log of realistic size.\n",
    ]
    for i in range(n):
        body.append(
            f"\n\ndef test_pipeline_invariant_{i:03d}():\n"
            f"    assert take_first(list(range(10)), 4) is not None\n"
        )
    return "".join(body)


def build_task(task_dir: Path, bug: Bug, n_filler: int = 380) -> dict:
    """Materialise one task on disk. Returns its manifest."""
    task_dir.mkdir(parents=True, exist_ok=True)
    if bug.correct not in MODULE:
        raise ValueError(f"bug {bug.key}: anchor line not found in module")

    (task_dir / "reportlib.py").write_text(MODULE.replace(bug.correct, bug.broken, 1))
    (task_dir / "test_reportlib.py").write_text(REAL_TESTS + _filler_tests(n_filler))
    (task_dir / "README.md").write_text(
        textwrap.dedent(
            """\
            # reportlib

            Utility library for the reporting pipeline. Run the suite with `pytest -q`.
            """
        )
    )
    return {
        "task_id": bug.key,
        "bug": bug.description,
        "broken_line": bug.broken,
        "correct_line": bug.correct,
        "files": ["reportlib.py", "test_reportlib.py"],
        "n_filler_tests": n_filler,
    }


def build_all(root: Path, n_filler: int = 380) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    return [build_task(root / bug.key, bug, n_filler) for bug in BUGS]


if __name__ == "__main__":
    import json
    import sys

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/eval-tasks")
    manifests = build_all(out)
    print(json.dumps(manifests, indent=2))
    print(f"\n{len(manifests)} tasks written to {out}", file=sys.stderr)
