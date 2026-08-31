"""The ``headroom output`` command group.

Registered through ``headroom.cli_extension``, which is the one seam that works
today without any core change — so the plugin is inspectable from the CLI the
moment it is installed, before the request-path seam it ultimately needs exists.

The commands exist to answer three questions an operator actually asks:

* *What is even running here?* — ``levers`` lists every lever with the wire
  formats and models it applies to, so a coverage gap is visible as a lever
  that matches nothing rather than as silence.
* *What would this do to my traffic?* — ``plan`` shapes a captured request body
  and prints the decision without sending anything. This is the artifact to
  hand a reviewer before enabling a lever.
* *Is it safe to turn on?* — ``shadow`` prints the bind rate: how often a
  lever's intended value would actually have constrained a real response. For
  the ceiling lever that number is the would-be truncation rate, and it is the
  number that decides whether it ships.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


@click.group("output")
def output_group() -> None:
    """Output-token shaping: inspect levers, dry-run plans, read shadow data."""


@output_group.command("levers")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def levers_cmd(as_json: bool) -> None:
    """List every registered lever and where it applies."""
    from .levers import build_default_levers

    rows: list[dict[str, Any]] = []
    for lever in build_default_levers():
        d = lever.descriptor
        rows.append(
            {
                "name": d.name,
                "summary": d.summary,
                "scope": "conversation" if d.mutates_cache_key else "turn",
                "wire_formats": [w.value for w in d.wire_formats],
                "models": list(d.model_prefixes) or ["*"],
                "turn_kinds": [k.value for k in d.turn_kinds] or ["*"],
                "requires_present": d.requires_present,
            }
        )

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return

    click.echo()
    click.echo(f"  {'LEVER':<16} {'SCOPE':<13} {'FORMATS':<34} REQUIRES")
    click.echo(f"  {'-' * 16} {'-' * 13} {'-' * 34} {'-' * 18}")
    for r in rows:
        fmts = ",".join(r["wire_formats"])
        req = r["requires_present"] or "—"
        click.echo(f"  {r['name']:<16} {r['scope']:<13} {fmts:<34} {req}")
    click.echo()
    for r in rows:
        click.echo(f"  {r['name']}: {r['summary']}")
    click.echo()


@output_group.command("plan")
@click.argument("body_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--harness", default="unknown", help="Which agent is wrapped (claude, codex, ...).")
@click.option("--verbosity", default=2, type=click.IntRange(0, 4), help="Steering level.")
@click.option("--input-tokens", default=0, type=int, help="Input token count, for stratification.")
def plan_cmd(body_file: Path, harness: str, verbosity: int, input_tokens: int) -> None:
    """Dry-run shaping over a captured request body. Sends nothing, writes nothing.

    BODY_FILE is a JSON provider request — an Anthropic Messages body, an
    OpenAI chat body, or a Responses payload. The format is detected.
    """
    from .shaper import build_default_shaper

    try:
        original = json.loads(body_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"could not read {body_file}: {exc}") from exc
    if not isinstance(original, dict):
        raise click.ClickException("request body must be a JSON object")

    body = json.loads(json.dumps(original))  # shape a copy; leave the file's meaning intact
    shaper = build_default_shaper(verbosity_level=verbosity)
    outcome = shaper.shape(body, harness=harness, input_tokens=input_tokens)

    click.echo()
    click.echo(f"  stratum   {outcome.stratum}")
    click.echo(f"  arm       {outcome.arm}")
    click.echo(f"  considered {', '.join(outcome.considered) or '(none)'}")
    if not outcome.considered:
        click.echo()
        click.echo(
            "  No lever matched this request. That is a coverage gap, not a"
            " no-op — this traffic would receive zero shaping."
        )
    if outcome.plan is not None and outcome.plan.skipped:
        click.echo(f"  skipped   {', '.join(outcome.plan.skipped)}")
    if outcome.plan is not None and outcome.plan.shadow_only:
        click.echo(f"  shadow    {', '.join(outcome.plan.shadow_only)}")
    click.echo()
    if outcome.changed:
        click.echo("  Would change:")
        for label in outcome.labels:
            if not label.startswith(("stratum:", "control:")):
                click.echo(f"    {label}")
    else:
        click.echo("  Would change: nothing")
    click.echo()


@output_group.command("shadow")
@click.argument("ledger_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--lever", default=None, help="Restrict to one lever.")
def shadow_cmd(ledger_file: Path, lever: str | None) -> None:
    """Report bind rates from a shadow ledger — the go/no-go number for a lever."""
    from .shadow import ShadowLedger

    try:
        ledger = ShadowLedger.load(ledger_file)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"could not read {ledger_file}: {exc}") from exc

    summary = ledger.summary()
    if not summary:
        click.echo("\n  No shadow records yet.\n")
        return

    click.echo()
    click.echo("  Bind rate = how often the intended value would have constrained")
    click.echo("  the real response. For a ceiling, this is the truncation rate.")
    click.echo()
    for key, row in sorted(summary.items()):
        if lever and not str(key).startswith(lever):
            continue
        click.echo(f"  {key}")
        for field_name, value in sorted(row.items()):
            if isinstance(value, float):
                click.echo(f"    {field_name:<16} {value:.3f}")
            else:
                click.echo(f"    {field_name:<16} {value}")
        click.echo()


def register(main: Any) -> None:
    """Entry point for ``headroom.cli_extension``.

    The core calls this with its root ``click.Group``. Adding a command group
    cannot silently change what an existing command does, which is why this
    seam treats installation itself as the opt-in.
    """
    main.add_command(output_group)
