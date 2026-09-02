"""
Demo: what headroom actually does to realistic agent context.

Builds three blobs of the kind a coding agent really reads, runs each through
headroom's compress(), and prints before/after token counts.

    python experiments/demo_compress.py
"""
import json
import random


random.seed(0)


def pytest_log(n_pass: int = 400) -> str:
    """A test run: a wall of PASSED lines plus one real failure."""
    lines = ["============================= test session starts =============================",
             "platform linux -- Python 3.10.12, pytest-8.0.0, pluggy-1.4.0",
             "rootdir: /repo", "collected %d items" % (n_pass + 1), ""]
    for i in range(n_pass):
        lines.append(f"tests/test_module_{i // 40}.py::test_case_{i} PASSED   [{100 * i // n_pass:3d}%]")
    lines += [
        "tests/test_billing.py::test_proration FAILED                          [100%]",
        "",
        "=================================== FAILURES ===================================",
        "________________________________ test_proration ________________________________",
        "",
        "    def test_proration():",
        "        inv = compute_invoice(days=15, monthly=100.0)",
        ">       assert inv.total == 50.0",
        "E       assert 48.387096774193544 == 50.0",
        "E        +  where 48.387096774193544 = Invoice(total=48.387096774193544).total",
        "",
        "tests/test_billing.py:42: AssertionError",
        "=========================== 1 failed, %d passed in 3.21s ======================" % n_pass,
    ]
    return "\n".join(lines)


def github_json(n: int = 120) -> str:
    """A tool call returning a big, highly repetitive JSON array."""
    return json.dumps([
        {
            "id": 1000 + i,
            "number": i,
            "title": f"Bug in module {i % 17}",
            "state": random.choice(["open", "closed"]),
            "user": {"login": f"dev{i % 9}", "id": i % 9, "type": "User",
                     "site_admin": False, "avatar_url": f"https://avatars.example/u/{i % 9}"},
            "labels": [{"id": 7, "name": "bug", "color": "d73a4a", "default": True}],
            "created_at": "2026-08-%02dT10:00:00Z" % (i % 28 + 1),
            "updated_at": "2026-08-%02dT12:00:00Z" % (i % 28 + 1),
            "comments": i % 5,
            "body": "Steps to reproduce: run the thing, observe the failure.",
        }
        for i in range(n)
    ], indent=2)


def source_file() -> str:
    """A source file the agent read in full to answer a narrow question."""
    body = []
    for i in range(30):
        body.append(f'''
def helper_{i}(payload: dict, *, strict: bool = False) -> dict:
    """Normalize payload variant {i} into the canonical shape."""
    result = {{}}
    for key, value in payload.items():
        if value is None and strict:
            raise ValueError(f"missing {{key}}")
        result[key.lower().strip()] = value
    result["_variant"] = {i}
    return result
''')
    return "import json\nimport logging\n\nlogger = logging.getLogger(__name__)\n" + "".join(body)


