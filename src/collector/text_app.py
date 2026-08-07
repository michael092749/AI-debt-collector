"""Text harness.

The same agent, the same engine, the same guardrails, with a terminal where the
phone would be. No audio, no keys, no network: if a negotiation misbehaves this
is where to reproduce it, and what reproduces here reproduces in the voice
worker because only the transport differs.

    uv run collector-text                  # negotiate, mock model, logging on
    uv run collector-text --verbose        # show the engine calls behind each turn
    uv run collector-text --claude         # same call, real model
    uv run collector-text --openrouter     # same call, real model via OpenRouter
    uv run collector-text --livekit        # same call, Gemini 3.6 Flash via LiveKit Inference
    uv run collector-agreements            # dump agreement records as JSON
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from collector.agent import AgentTurn, CallReport, NegotiationAgent
from collector.audit.events import dumps
from collector.audit.store import AuditStore, default_db_path
from collector.guardrails.rings import PreCallContext
from collector.llm.base import LLMClient
from collector.llm.mock_client import MockLLMClient
from collector.policy import PolicyConfig
from collector.tracing import configure_tracing, flush_traces

PROMPT = "you> "


def _client(use_claude: bool, use_openrouter: bool, use_livekit: bool) -> LLMClient:
    """The mock unless asked otherwise. The real clients are imported late and
    on purpose: the default path must work with no key, no network and no SDK
    installed."""
    if sum((use_claude, use_openrouter, use_livekit)) > 1:
        raise SystemExit("--claude, --openrouter and --livekit are mutually exclusive")
    if use_livekit:
        try:
            from collector.llm.livekit_client import LiveKitInferenceClient
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit(
                "--livekit needs the openai SDK, livekit-agents, and "
                "LIVEKIT_API_KEY/LIVEKIT_API_SECRET; run without it to use the "
                f"scripted client ({exc})"
            ) from exc
        return LiveKitInferenceClient()
    if use_openrouter:
        try:
            from collector.llm.openrouter_client import OpenRouterClient
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit(
                "--openrouter needs the openai SDK and OPENROUTER_API_KEY; "
                f"run without it to use the scripted client ({exc})"
            ) from exc
        client: LLMClient = OpenRouterClient()
        return client
    if not use_claude:
        return MockLLMClient()
    try:
        from collector.llm.anthropic_client import AnthropicClient
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "--claude needs the anthropic SDK and ANTHROPIC_API_KEY; "
            f"run without it to use the scripted client ({exc})"
        ) from exc
    return AnthropicClient()


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collector-text", description="Negotiate with the collections agent in a terminal."
    )
    parser.add_argument("--name", default="Dana Whitfield", help="Consumer name for the call.")
    parser.add_argument("--account", default="ACCT-4471", help="Account reference.")
    parser.add_argument("--call-id", default="call-text-1", help="Call id used in the log.")
    parser.add_argument("--db", type=Path, default=default_db_path(), help="SQLite path.")
    parser.add_argument("--no-store", action="store_true", help="Run without writing a log.")
    parser.add_argument("--claude", action="store_true", help="Use the real model.")
    parser.add_argument(
        "--openrouter", action="store_true", help="Use the real model via OpenRouter."
    )
    parser.add_argument(
        "--livekit", action="store_true", help="Use Gemini 3.6 Flash via LiveKit Inference."
    )
    parser.add_argument("--verbose", action="store_true", help="Show engine calls and guard trips.")
    args = parser.parse_args(argv)

    # Before the store, so a malformed COLLECTOR_TRACING is a start-up failure
    # rather than a call that quietly exports nothing (see tracing.py).
    configure_tracing()

    store = None if args.no_store else AuditStore(args.db, json_dir=args.db.parent)
    try:
        return _negotiate(args, store)
    finally:
        if store is not None:
            store.close()
        flush_traces()


def _negotiate(args: argparse.Namespace, store: AuditStore | None) -> int:
    agent = NegotiationAgent(
        llm=_client(args.claude, args.openrouter, args.livekit),
        policy=PolicyConfig.default(),
        call_id=args.call_id,
        consumer_name=args.name,
        account_ref=args.account,
        store=store,
    )

    check, opening = agent.open_call(
        PreCallContext(account_loaded=True, within_calling_window=True)
    )
    if opening is None:
        print("Call not placed. Pre-call guardrails blocked it:")
        for violation in check.violations:
            print(f"  - {violation.detail}")
        return 1

    print(f"\nagent> {opening}\n")
    print("(ctrl-d or an empty line ends the call)\n")

    while not agent.ended:
        try:
            said = input(PROMPT).strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if not said:
            break

        turn = agent.turn(said)
        if args.verbose:
            _trace(turn)
        print(f"\nagent> {turn.spoken or '(nothing said — the turn was held back)'}\n")

    report = agent.close()
    _summarize(report, store)
    return 0


def _trace(turn: AgentTurn) -> None:
    """Show what happened behind the sentence: which tools ran, what the engine
    ruled, and anything the guard held back."""
    for result in turn.tool_results:
        outcome = result.payload.get("outcome") or ("ok" if result.ok else "refused")
        detail = result.payload.get("rationale_code") or result.payload.get("error") or ""
        print(f"  [engine] {result.name}: {outcome} {detail}".rstrip())
        if result.verdict is not None:
            for condition in result.verdict.conditions:
                mark = "pass" if condition.passed else "FAIL"
                print(
                    f"           {mark} {condition.rule_id}: {condition.actual} {condition.limit}"
                )
    for blocked in turn.blocked:
        print(f"  [guard]  held back before TTS: {blocked!r}")


def _summarize(report: CallReport, store: AuditStore | None) -> None:
    outcome = report.outcome
    summary = report.summary
    offer = report.agreed_offer

    print("-" * 72)
    print(f"outcome:     {outcome.value}")
    print(f"compliant:   {summary.compliant}")
    print(f"turns:       {report.turns}   blocked: {summary.blocked_turns}")
    print(f"disclosures: {', '.join(d.value for d in summary.disclosures_fired) or 'none'}")
    escalation = summary.escalation
    if escalation is not None:
        print(f"escalation:  {escalation.trigger}")
        # The obligation the call ended owing. It is on the audit trail either
        # way; printing it is what makes the commitment checkable from the
        # terminal, with no phone and no human queue in the loop.
        if escalation.callback_owed:
            print(f"callback:    owed — a human owes a call back (turn {escalation.turn_index})")
    if offer is not None:
        schedule = ", ".join(f"{i.amount} on day {i.due_day_offset}" for i in offer.installments)
        print(f"agreement:   {offer.tier.label} — {offer.total} ({offer.cadence.value})")
        print(f"schedule:    {schedule}")
    if store is not None:
        record = store.agreement(report.call_id)
        if record is not None:
            print(f"record:      {store.agreement_json_path(record.agreement_id)}")


def dump_agreements(argv: list[str] | None = None) -> int:
    """``collector-agreements`` — every agreement with its decision trail."""
    parser = argparse.ArgumentParser(
        prog="collector-agreements", description="Dump agreement records as JSON."
    )
    parser.add_argument("--db", type=Path, default=default_db_path())
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"no log at {args.db}", file=sys.stderr)
        return 1
    with AuditStore(args.db) as store:
        records = store.agreements()
        print(dumps(list(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
