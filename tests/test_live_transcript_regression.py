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
from collector.guardrails.disclosures import MINI_MIRANDA_TEXT
from collector.guardrails.rings import CONNECTIVE_TEXT, PreCallContext
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
