"""Detection of test-runner output (pytest/go/cargo/jest/JUnit) as BUILD_OUTPUT.

Regression cover for the gap where ``_LOG_PATTERNS`` anchored test status to the
start of a line (``^\\s*PASSED``). No mainstream runner emits it there, so a
400-line pytest log scored a 0.005 pattern density against the 0.10 threshold and
fell through to PLAIN_TEXT — which routed it to generic text compression instead of
LogCompressor.
"""

import pytest

from headroom.transforms.content_detector import ContentType, detect_content_type


def _pytest_log(n_pass: int = 400) -> str:
    lines = [
        "============================= test session starts =============================",
        "platform linux -- Python 3.13.0, pytest-8.0.0, pluggy-1.4.0",
        "rootdir: /repo",
        f"collected {n_pass + 1} items",
        "",
    ]
    lines += [
        f"tests/test_module_{i // 40}.py::test_case_{i} PASSED   [{100 * i // n_pass:3d}%]"
        for i in range(n_pass)
    ]
    lines += [
        "tests/test_billing.py::test_proration FAILED                          [100%]",
        "",
        "=================================== FAILURES ===================================",
        "    def test_proration():",
        ">       assert inv.total == 50.0",
        "E       assert 48.387096774193544 == 50.0",
        "tests/test_billing.py:42: AssertionError",
        f"=========================== 1 failed, {n_pass} passed in 3.21s ================",
    ]
    return "\n".join(lines)


def _go_test_log() -> str:
    return "\n".join(
        [f"=== RUN   TestFoo{i}" for i in range(20)]
        + [f"--- PASS: TestFoo{i} (0.00s)" for i in range(20)]
        + ["ok  \tgithub.com/x/y\t0.012s"]
    )


def _cargo_test_log() -> str:
    return "\n".join(
        [f"test tests::case_{i} ... ok" for i in range(30)]
        + ["test result: FAILED. 29 passed; 1 failed; 0 ignored"]
    )


def _jest_log() -> str:
    return "\n".join(
        [f"  ✓ renders row {i} (12 ms)" for i in range(25)]
        + ["Tests:       1 failed, 24 passed, 25 total"]
    )


@pytest.mark.parametrize(
    "name,content",
    [
        ("pytest", _pytest_log()),
        ("go test", _go_test_log()),
        ("cargo test", _cargo_test_log()),
        ("jest", _jest_log()),
    ],
    ids=["pytest", "go", "cargo", "jest"],
)
def test_test_runner_output_detected_as_build(name, content):
    result = detect_content_type(content)
    assert result.content_type is ContentType.BUILD_OUTPUT, (
        f"{name} output classified as {result.content_type} — it will not reach "
        f"LogCompressor"
    )
    assert result.confidence >= 0.5


def test_failure_tail_is_scored_even_in_a_long_log():
    """Detection samples head AND tail: the diagnostic payload is back-loaded."""
    result = detect_content_type(_pytest_log(n_pass=3000))
    assert result.content_type is ContentType.BUILD_OUTPUT
    assert result.metadata["test_matches"] > 0


def test_prose_mentioning_test_counts_is_not_build_output():
    """English prose says 'the test passed'; a runner says 'PASSED'."""
    prose = (
        "The migration went well. We ran the suite and 3 passed on the first try.\n"
        "Nothing else of note happened this week; the team is happy with progress.\n"
    ) * 20
    assert detect_content_type(prose).content_type is not ContentType.BUILD_OUTPUT


def test_source_code_is_not_reclassified_as_build_output():
    src = (
        "import json\n\n"
        + "".join(
            f"def helper_{i}(payload: dict) -> dict:\n"
            f'    """Normalize variant {i}."""\n'
            f"    return {{k.lower(): v for k, v in payload.items()}}\n\n"
            for i in range(30)
        )
    )
    assert detect_content_type(src).content_type is ContentType.SOURCE_CODE


def test_a_single_status_line_is_not_enough():
    """The >=3 absolute floor keeps one stray status line from flipping a document."""
    doc = "Release notes\n\n" + "Some ordinary prose here.\n" * 40 + "check PASSED\n"
    assert detect_content_type(doc).content_type is not ContentType.BUILD_OUTPUT
