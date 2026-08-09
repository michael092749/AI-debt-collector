"""The 2026-08-07 live call, replayed end to end.

Provenance: a real Gemini-route call against ``PolicyConfig.default()`` —
$1,000 balance, $250 minimum payment, $800 settlement floor, the four-tier
ladder PAY_IN_FULL -> DOWNPAYMENT_PLUS_ONE -> SETTLEMENT -> PAYMENT_PLAN. The
consumer (Dana Whitfield) confirmed identity, refused the full balance, named
$150 and then $300, and the call ran out of road at the second tier of four.

Three separate defects from that call are covered elsewhere (the repeated
Mini-Miranda, the unrelayed rationale, the dropped capacity signal). This file
covers what is left once those are fixed, and none of it is phrasing: the round
budget cannot pay for the ladder, the bottom tier is unreachable by conceding,
a rejection is explained by a rule that did not reject it, and two of the
agent's sentences were authored by the guard rather than the model.

The model is scripted rather than mocked. ``MockLLMClient`` is a compliant
stand-in by construction — it reads figures out of ``you_may_say`` and never
mislabels an offer — so it cannot reproduce a call that went wrong. What is
under test here is the system's response to a model that behaves the way this
one did, so the model's side of the call is replayed verbatim.

These are reproduce-red regression tests. They are expected to fail at
HEAD; each one names the property that should hold once the defect is fixed.
"""

from __future__ import annotations

from collector.agent import NegotiationAgent
from collector.decision_engine import _HARD_FLOORS, RationaleCode, validate_offer
from collector.guardrails.disclosures import MINI_MIRANDA_TEXT, fires_mini_miranda
from collector.guardrails.rings import CONNECTIVE_TEXT, SAFE_FALLBACK_TEXT, PreCallContext
from collector.llm.base import LLMResponse, Message, ToolCall
from collector.money import Money
from collector.offers import Cadence, ConsumerProposal, Tier
from collector.policy import PolicyConfig
from collector.tools import TOOL_SCHEMAS, ToolContext, execute

POLICY = PolicyConfig.default()

_OPENING = (
    "This is an AI calling on behalf of Meridian Recovery Services. A real "
    "person is available to speak with you if you prefer. Am I speaking with "
    "Dana Whitfield?"
)

# The turn the agent actually spoke about the $250/$750 arrangement. "payment
# plan" is the label of Tier.PAYMENT_PLAN — a different, worse tier than the
# DOWNPAYMENT_PLUS_ONE the engine had just authored.
_MISLABELLED_OFFER = (
    "Okay. I can offer a payment plan for the one thousand dollars. It would "
    "be two monthly payments, starting with two hundred fifty dollars today "
    "and then seven hundred fifty dollars in thirty days."
)


class _ScriptedClient:
    """Replays one side of a real call. Deterministic, and holds no state.

    Anything past the end of the script is a filler line, so a test that
    stops caring after turn N does not have to script turn N+1.
    """

    def __init__(self, *script: LLMResponse) -> None:
        self._script = list(script)

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        return self._script.pop(0) if self._script else LLMResponse(text="Okay.")


def _agent(*script: LLMResponse) -> NegotiationAgent:
    agent = NegotiationAgent(
        llm=_ScriptedClient(*script),
        policy=POLICY,
        call_id="live-2026-08-07",
        consumer_name="Dana Whitfield",
    )
    agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
    return agent


def _tool(name: str, **arguments: object) -> LLMResponse:
    return LLMResponse(tool_calls=(ToolCall(name=name, arguments=arguments),))


# --------------------------------------------------------------------------
# The round budget
# --------------------------------------------------------------------------


def _replay_the_tool_calls() -> ToolContext:
    """The engine-visible half of the live call, in the order it happened.

    Reconstructed from the offers the agent read out: the $250/$750 split it
    described is what ``build_counter`` produces for DOWNPAYMENT_PLUS_ONE at a
    $150 capacity, and nothing else in the tool surface produces it.
    """
    context = ToolContext.opening(POLICY)
    for call in (
        # "There is an overdue balance of one thousand dollars ... in full today?"
        ToolCall(name="propose_offer", arguments={"preferred_cadence": "immediate"}),
        # "Hundred and fifty dollars." relayed as a per-payment figure.
        ToolCall(
            name="validate_consumer_offer",
            arguments={
                "amount_each": "150",
                "payment_count": 7,
                "cadence": "monthly",
                "signaled_capacity": "150",
            },
        ),
        # "I can only pay a hundred and fifty dollars."
        ToolCall(
            name="validate_consumer_offer",
            arguments={"total": "150", "payment_count": 1, "cadence": "immediate"},
        ),
        # "No." -> the concession that produced $250 today / $750 in thirty days.
        ToolCall(name="record_refusal"),
        ToolCall(name="concede", arguments={"preferred_cadence": "monthly"}),
        # "No. I can only do three hundred dollars."
        ToolCall(
            name="validate_consumer_offer",
            arguments={"total": "300", "payment_count": 1, "cadence": "immediate"},
        ),
    ):
        context = execute(call, context).context
    return context


def test_the_round_budget_can_pay_for_the_whole_ladder() -> None:
    """The call must be able to reach the bottom of the ladder it is given.

    The live call spent 5 of its 8 rounds reaching the second tier of four,
    leaving 3 for the two below it — and the original form of this test measured
    exactly that, by replaying the transcript's tool calls verbatim (see
    ``_replay_the_tool_calls``, kept below for the other assertions).

    That measurement is no longer the right one, because the sequence it replays
    can no longer occur. Two of those five rounds were the same $150 ruled on
    twice, and the sixth call was a `concede` that a refusal now buys on its
    own. What matters is not that one historical sequence fits the budget but
    that a consumer naming a figure they can actually fund reaches an
    arrangement built on it before the cap closes the call — so that is what is
    asserted, against the flow the engine now produces.

    The cap is anti-badgering and is deliberately not raised to accommodate a
    wasteful sequence: the fix for "the ladder ran out of road" is to stop
    spending rounds on nothing, not to license more haggling.
    """
    context = ToolContext.opening(POLICY)
    reached: list[Tier] = []
    # "I can only do three hundred dollars", said and held to — the position
    # Dana was in when the live call stalled at $250/$750.
    while not context.state.is_exhausted:
        result = execute(
            ToolCall(
                name="validate_consumer_offer",
                arguments={"amount_each": "300", "cadence": "monthly"},
            ),
            context,
        )
        context = result.context
        if result.offer is not None:
            reached.append(result.offer.tier)
        if result.payload.get("outcome") == "accept":
            break

    assert Tier.PAYMENT_PLAN in reached, (
        f"a consumer who can fund $300 a month never reached a payment plan in "
        f"{POLICY.max_negotiation_rounds} rounds: {[t.label for t in reached]}"
    )
    assert context.state.round_count <= POLICY.max_negotiation_rounds


def test_every_tier_on_the_ladder_can_be_put_on_the_table() -> None:
    """``concede`` is the only way the agent may move down the ladder, so every
    tier the policy defines has to be reachable through it.

    ``_is_concession`` compares totals, and a payment plan is the full balance
    while a settlement is $800 — so the step from SETTLEMENT to PAYMENT_PLAN
    never reads as giving ground and never happens. The bottom tier is dead.
    """
    context = ToolContext.opening(POLICY)
    context = execute(
        ToolCall(
            name="validate_consumer_offer",
            arguments={"total": "150", "payment_count": 1, "cadence": "immediate"},
        ),
        context,
    ).context

    offered: set[Tier] = {Tier.PAY_IN_FULL}
    for _ in range(POLICY.max_negotiation_rounds):
        context = execute(ToolCall(name="record_refusal"), context).context
        result = execute(
            ToolCall(name="concede", arguments={"preferred_cadence": "monthly"}), context
        )
        context = result.context
        if result.offer is not None:
            offered.add(result.offer.tier)

    assert offered == set(Tier), f"never put on the table: {sorted(set(Tier) - offered)}"


# A third round-budget defect is deliberately not asserted here. Once the
# ladder bottoms out, ``concede`` returns ``moved: False`` and still charges a
# round, so two dead concessions can burn the last of the budget — but
# ``_concede`` records that round on purpose (tools.py), to keep the cap
# reachable for a consumer who refuses forever without naming a figure. That is
# a real tension, not a plain bug, and the fix belongs upstream: make
# PAYMENT_PLAN reachable and the concession stops being dead. See the test
# above.


# --------------------------------------------------------------------------
# The rationale the consumer hears
# --------------------------------------------------------------------------


def test_a_rejection_is_explained_by_a_rule_that_could_have_rejected_it() -> None:
    """The consumer offered $150 and was told it was more than they owed.

    ``$150 x 7`` is $1,050, so NO_OVER_COLLECTION fails — but so does
    MIN_PAYMENT, and MIN_PAYMENT is the hard floor that actually made the
    outcome a rejection. ``_first_failure`` walks the conditions in declaration
    order and returns the first one it finds, which puts a soft rule's name on
    a hard rule's verdict.
    """
    # $150 x 7 — the total the engine derives from the figure the consumer
    # named and the payment count the model had to invent to relay it.
    proposal = ConsumerProposal(
        total=Money("1050.00"),
        payment_count=7,
        cadence=Cadence.MONTHLY,
        signaled_capacity=Money("150.00"),
    )
    verdict = validate_offer(proposal, ToolContext.opening(POLICY).state, POLICY)

    assert verdict.outcome == "reject"
    deciding = {c.rule_id for c in verdict.conditions if not c.passed} & _HARD_FLOORS
    assert deciding, "a rejection with no hard floor failing"
    assert verdict.rationale_code in {
        RationaleCode.BELOW_SETTLEMENT_FLOOR,
        RationaleCode.BELOW_MIN_PAYMENT,
        RationaleCode.DISCOUNT_NOT_AUTHORIZED,
    }, (
        f"rejected on {sorted(r.value for r in deciding)} but explained as "
        f"{verdict.rationale_code.value}"
    )


def test_over_collection_is_never_the_reason_a_lowball_is_refused() -> None:
    """The same defect stated as the sentence it produced.

    Whatever payment count the model picks, a consumer who names a figure
    below the minimum payment must not be told their offer was too large.
    """
    for payment_count in range(1, POLICY.max_installments + 4):
        result = execute(
            ToolCall(
                name="validate_consumer_offer",
                arguments={
                    "amount_each": "150",
                    "payment_count": payment_count,
                    "cadence": "monthly",
                    "signaled_capacity": "150",
                },
            ),
            ToolContext.opening(POLICY),
        )
        assert result.payload["rationale_code"] != RationaleCode.ABOVE_BALANCE_OWED.value, (
            f"$150 x {payment_count} refused as above the balance owed"
        )


def test_a_bare_capacity_can_be_relayed_without_inventing_a_schedule() -> None:
    """ "What can you manage?" - "A hundred and fifty dollars."

    That is a capacity, not a proposal. ``validate_consumer_offer`` requires
    ``payment_count`` and ``cadence``, so the model has to invent both before
    the engine will hear the one figure the consumer actually said — and the
    invented count is what turned $150 into a $1,050 over-collection.
    """
    schema = next(s for s in TOOL_SCHEMAS if s.name == "validate_consumer_offer")
    required = {p.name for p in schema.params if p.required}

    assert not required & {"payment_count", "cadence"}, (
        "relaying a bare capacity forces the model to invent "
        f"{sorted(required & {'payment_count', 'cadence'})}"
    )


# --------------------------------------------------------------------------
# The sentences the guard wrote
# --------------------------------------------------------------------------


def test_the_mini_miranda_is_not_something_a_connective_can_close() -> None:
    """The wasted turn, reproduced.

    The notice went out, the sentence behind it was blocked, and
    ``_stands_on_its_own`` ruled that a completed Mini-Miranda is enough for
    ``CONNECTIVE_TEXT`` to close. It is not: the notice is a disclosure, not a
    proposition, so "Does that work for you?" refers to nothing. The same
    docstring already rejects this reasoning for a sentence that merely
    announces an offer.
    """
    agent = _agent(
        LLMResponse(text=_OPENING),
        # An amount the engine never authored, so the second sentence blocks
        # and the turn is left holding only the notice.
        LLMResponse(
            text=(
                f"{MINI_MIRANDA_TEXT} There is an overdue balance of four "
                "hundred twelve dollars on your account."
            )
        ),
    )

    spoken = list(agent.stream_turn("Yes."))

    assert any(MINI_MIRANDA_TEXT in line for line in spoken)
    assert CONNECTIVE_TEXT not in spoken, (
        f"the notice was closed with a dangling question: {spoken}"
    )


def test_an_offer_is_never_described_with_another_tiers_name() -> None:
    """The agent called a DOWNPAYMENT_PLUS_ONE arrangement a "payment plan".

    ``_offer_payload`` hands the model ``tier: "downpayment plus one"`` and
    nothing anywhere checks what the model calls it back. "payment plan" is the
    label of Tier.PAYMENT_PLAN — four payments of $250, an arrangement this
    consumer was never offered and (see the ladder tests above) could not have
    been. The audit record and the transcript disagree about what was offered,
    which is the one thing the decision record exists to prevent.
    """
    agent = _agent(
        LLMResponse(text=_OPENING),
        _tool("propose_offer", preferred_cadence="immediate"),
        LLMResponse(
            text=(
                f"{MINI_MIRANDA_TEXT} There is an overdue balance of one thousand "
                "dollars on your account. Can you clear that in full today?"
            )
        ),
        _tool(
            "validate_consumer_offer",
            amount_each="150",
            payment_count=7,
            cadence="monthly",
            signaled_capacity="150",
        ),
        LLMResponse(
            text=(
                "I cannot accept that because it would result in paying more than "
                "the total balance owed. Would you be able to pay the full one "
                "thousand dollars today?"
            )
        ),
        LLMResponse(
            tool_calls=(
                ToolCall(name="record_refusal"),
                ToolCall(name="concede", arguments={"preferred_cadence": "monthly"}),
            )
        ),
        LLMResponse(text=_MISLABELLED_OFFER),
    )

    for said in ("Yes.", "No.", "Hundred and fifty dollars.", "No."):
        agent.turn(said)

    standing = agent.tools.standing_offer
    assert standing is not None
    assert standing.tier is Tier.DOWNPAYMENT_PLUS_ONE

    spoken = " ".join(agent.spoken).lower()
    mislabels = {tier.label for tier in Tier if tier is not standing.tier and tier.label in spoken}
    assert not mislabels, f"a {standing.tier.label} offer was described as: {sorted(mislabels)}"


# --------------------------------------------------------------------------
# The 2026-08-08 call: an accepted arrangement nobody said out loud
# --------------------------------------------------------------------------

# What the agent actually said on the turn the engine accepted $400/$600. Every
# figure of the deal is in ``you_must_confirm``; none of them is in the sentence.
_SILENT_ACCEPT = (
    "That works. So that'll be a payment today and one more after it. Does that work for you?"
)


def _through_the_concession(final: LLMResponse) -> NegotiationAgent:
    """The live call up to the turn that accepted the consumer's own split.

    "I can pay four hundred today, and then I can pay six hundred next month"
    against a conceded DOWNPAYMENT_PLUS_ONE — the engine returns ``accept`` on
    a $400/$600 schedule and hands the model the line naming both figures.
    """
    split = _tool(
        "validate_consumer_offer",
        total="1000",
        payment_count=2,
        cadence="monthly",
        signaled_capacity="400",
    )
    agent = _agent(
        LLMResponse(text=_OPENING),
        _tool("propose_offer", preferred_cadence="immediate"),
        LLMResponse(
            text=f"{MINI_MIRANDA_TEXT} The balance is one thousand dollars. Can you clear it today?"
        ),
        # Their split, ruled on while the ladder still stands at pay-in-full:
        # a counter, and the $400 goes on the record as their capacity.
        split,
        LLMResponse(text="I can't set that up as things stand. Is the full amount possible?"),
        # The refusal buys the step down, and the concession is built on the
        # capacity the first ruling recorded — $400 today, $600 in thirty days.
        LLMResponse(
            tool_calls=(
                ToolCall(name="record_refusal"),
                ToolCall(name="concede", arguments={"preferred_cadence": "monthly"}),
            )
        ),
        split,
        final,
    )
    for said in (
        "Yes.",
        "I can pay four hundred today, and then I can pay six hundred next month.",
        "No, that's really all I can manage.",
    ):
        agent.turn(said)
    return agent


def test_an_accepted_arrangement_is_read_back_with_its_figures() -> None:
    """ "That works. So that'll be a payment today and one more after it." —
    and the consumer had to ask "So what's the deal?" to hear the numbers
    (live call, 2026-08-08).

    The turn is not a guard failure: it states no figure, so nothing is
    unauthorized and it clears the pre-TTS check untouched. It is a *silence*
    failure, and the guard has no rule that fires on what a sentence leaves out.

    ``you_must_confirm`` carries the whole schedule and the model is told to say
    it. On the one turn where the terms become an obligation, "told to" is not a
    control — the deal the consumer just agreed to has to be the deal they
    heard, so the figures are ours to supply the way the Mini-Miranda is.
    """
    agent = _through_the_concession(LLMResponse(text=_SILENT_ACCEPT))

    spoken = agent.turns[-1].spoken or ""
    assert "400" in spoken and "600" in spoken, (
        f"the engine accepted $400/$600 and the consumer heard: {spoken!r}"
    )


def test_a_read_back_the_model_wrote_itself_is_left_alone() -> None:
    """The repair is for silence, not for phrasing. A turn that already names
    the schedule must reach TTS as the model wrote it — appending a second
    canonical read-back would have the consumer hear the terms twice, which is
    the stutter that killed the same shape of fix for MINI_MIRANDA_OUT_OF_ORDER.
    """
    said = (
        "That works. Just to confirm: $400.00 today, then $600.00 in 30 days "
        "— $1,000.00 in total. Should I set that up?"
    )
    agent = _through_the_concession(LLMResponse(text=said))

    assert agent.turns[-1].spoken == said


def test_the_streaming_path_reads_the_agreement_back_too() -> None:
    """The repair has to live on the path that carries real calls.

    ``turn()`` guards a finished paragraph and can rewrite it; ``stream_turn``
    puts each sentence on the wire as it completes and cannot take one back. So
    the read-back cannot be spliced into the candidate here — it is spoken after
    the model has finished, as its own chunk, which is also the only ordering
    that makes sense out loud: acknowledge, then state the terms.

    Fixing only ``turn()`` would leave production exactly as broken as the
    transcript that prompted this, and every test above would still pass — the
    same gap ``test_the_streaming_path_can_satisfy_the_gate`` was written for.
    """
    split = _tool(
        "validate_consumer_offer",
        total="1000",
        payment_count=2,
        cadence="monthly",
        signaled_capacity="400",
    )
    agent = _agent(
        LLMResponse(text=_OPENING),
        _tool("propose_offer", preferred_cadence="immediate"),
        LLMResponse(
            text=f"{MINI_MIRANDA_TEXT} The balance is one thousand dollars. Can you clear it today?"
        ),
        split,
        LLMResponse(text="I can't set that up as things stand. Is the full amount possible?"),
        LLMResponse(
            tool_calls=(
                ToolCall(name="record_refusal"),
                ToolCall(name="concede", arguments={"preferred_cadence": "monthly"}),
            )
        ),
        split,
        LLMResponse(text=_SILENT_ACCEPT),
    )

    for said in (
        "Yes.",
        "I can pay four hundred today, and then I can pay six hundred next month.",
        "No, that's really all I can manage.",
    ):
        list(agent.stream_turn(said))

    heard = agent.turns[-1].spoken or ""
    assert "400" in heard and "600" in heard, (
        f"the engine accepted $400/$600 and the streamed turn was: {heard!r}"
    )


def test_the_read_back_is_never_bolted_onto_a_scripted_recovery() -> None:
    """A turn that fell back is not a turn to append terms to.

    Observed in a live trial (2026-08-08): "I'd rather not misstate anything,
    so let me keep this simple. What would work for you? Just to confirm:
    $1,000.00 today. Should I set that up?" — the recovery line reopens the
    negotiation and the read-back closes it, in one breath, on a turn where the
    model had already lost the thread.

    The same reasoning ``_stream_connective`` gives for not spending
    ``SAFE_FALLBACK_TEXT`` over a good offer, in the other direction. The
    agreement stays owed; ``_accepted`` is not cleared, so the next turn that
    the model actually writes still carries it.
    """
    split = _tool(
        "validate_consumer_offer",
        total="1000",
        payment_count=2,
        cadence="monthly",
        signaled_capacity="400",
    )
    agent = _agent(
        LLMResponse(text=_OPENING),
        _tool("propose_offer", preferred_cadence="immediate"),
        LLMResponse(
            text=f"{MINI_MIRANDA_TEXT} The balance is one thousand dollars. Can you clear it today?"
        ),
        split,
        LLMResponse(text="I can't set that up as things stand. Is the full amount possible?"),
        LLMResponse(
            tool_calls=(
                ToolCall(name="record_refusal"),
                ToolCall(name="concede", arguments={"preferred_cadence": "monthly"}),
            )
        ),
        split,
        # Accepted terms, and then a turn that cannot be spoken: an invented
        # balance takes both strikes and the scripted line closes the turn.
        LLMResponse(text="Your balance is $1,573.97."),
        LLMResponse(text="Your balance is $1,573.97."),
        LLMResponse(text="Your balance is $1,573.97."),
    )

    for said in (
        "Yes.",
        "I can pay four hundred today, and then I can pay six hundred next month.",
        "No, that's really all I can manage.",
    ):
        list(agent.stream_turn(said))

    heard = agent.turns[-1].spoken or ""
    assert "400" not in heard and "600" not in heard, (
        f"a read-back was appended to a scripted recovery line: {heard!r}"
    )
    # Still owed, not written off: the next real turn has to carry it.
    assert agent._accepted is not None


def test_the_streaming_path_repairs_a_missing_mini_miranda_too() -> None:
    """ "Yes." answered with the scripted fallback, on the path that carries
    calls (live, 2026-08-09).

    ``8c41068`` put the canonical notice in front of a turn whose only fault is
    that it lacks one — in ``_guard_and_speak``, which is ``turn()``. Voice
    calls go through ``_guard_sentence``, which never had the repair, so
    production kept spending two strikes and the fallback on exactly the turn
    the fix was written for. Measured on the live route at the deployed commit:
    4 of 6 openings fell back.

    The model cannot be prompted out of this — the notice may not be quoted in
    the prompt (``test_no_required_script_reads_as_a_prompt_leak``), so it
    paraphrases, and "I'm calling about a debt you owe" does not contain the
    literal "attempt to collect a debt" the detector requires.
    """
    paraphrased = (
        "I'm calling about a debt you owe, and anything you tell me may be "
        "used to collect it. You have a balance of one thousand dollars. "
        "Can you clear it today?"
    )
    agent = _agent(
        LLMResponse(text=_OPENING),
        _tool("propose_offer", preferred_cadence="immediate"),
        # It paraphrases again on every rewrite, which is what spends the
        # strike budget and reaches the fallback.
        LLMResponse(text=paraphrased),
        LLMResponse(text=paraphrased),
        LLMResponse(text=paraphrased),
    )

    heard = "".join(agent.stream_turn("Yes."))

    assert SAFE_FALLBACK_TEXT not in heard, (
        f"the turn after identity confirmation fell back: {heard!r}"
    )
    assert fires_mini_miranda(heard), f"no Mini-Miranda reached the consumer: {heard!r}"


def test_a_repeated_notice_does_not_cost_the_whole_turn() -> None:
    """The fallback the consumer still hears occasionally (live, 2026-08-09).

    Once a notice is on record — the model's own, or the one the repair put
    there — a model that says it again trips ``MINI_MIRANDA_REDUNDANT``. That
    is not curable by more text, so the sentence blocks, and a block is what
    walks the turn to ``SAFE_FALLBACK_TEXT``. The consumer had already heard a
    complete, correctly-ordered disclosure; the turn was then thrown away for
    saying it twice.

    A duplicate notice carries no information the consumer has not had, so the
    sentence is dropped and the turn carries on. Blocking is for a sentence
    that must not be said; this is a sentence that need not be.
    """
    agent = _agent(
        LLMResponse(text=_OPENING),
        # A paraphrase with no figure in it: the repair supplies the notice,
        # and nothing has been said that a connective could close over. This
        # is the live shape — the recovery below is the fallback, not the
        # connective, precisely because no figure reached the consumer yet.
        LLMResponse(
            text="I'm calling about a debt you owe. Let me pull up your account.",
            tool_calls=(ToolCall(name="propose_offer", arguments={}),),
        ),
        LLMResponse(text=f"{MINI_MIRANDA_TEXT} You have a balance of one thousand dollars."),
    )

    heard = "".join(agent.stream_turn("Yes."))

    assert SAFE_FALLBACK_TEXT not in heard, f"a repeated notice cost the turn: {heard!r}"
    assert "one thousand dollars" in heard, f"the balance never reached the consumer: {heard!r}"
