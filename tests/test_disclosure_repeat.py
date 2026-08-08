"""The Mini-Miranda is said once, and saying it twice is not merely discouraged.

A live call on 2026-08-07 delivered the debt-collection notice on two
consecutive agent turns:

    A: "This is an AI calling on behalf of Meridian Recovery Services. A real
        person is available to speak with you if you prefer. Am I speaking
        with Dana Whitfield?"
    C: "Yes."
    A: "this is an attempt to collect a debt by a debt collector and any
        information obtained will be used for that purpose. Does that work
        for you?"
    C: "Yes."
    A: "This is an attempt to collect a debt by a debt collector and any
        information obtained will be used for that purpose. There is an
        overdue balance of one thousand dollars on your account. Can you
        clear that in full today?"

Two separate defects are in that trace and each gets its own test here:

1.  The notice is delivered twice. ``DisclosureState`` latches
    ``MINI_MIRANDA`` correctly on the first delivery — that half works — but
    nothing reads the latch on the way *out*. ``check_outbound`` has no rule
    against a redundant notice, so a second delivery is spoken to the
    consumer with the guard's full approval. The prompt asks the model not to
    repeat it (``SYSTEM_PROMPT``: "Say it once, in full. After that it is
    behind you and you do not repeat it") and, four lines later, tells it to
    lead with the notice again whenever a turn is stopped. The model obeyed
    the second instruction. A compliance obligation that rests on which of
    two conflicting prompt lines the model happens to weight is not a
    control.

2.  The first delivery wasted the turn. The notice sentence cleared the
    guard; the balance sentence behind it was blocked as an unauthorized
    amount; and ``_stands_on_its_own`` then treated the completed notice as
    something a connective could close, so the turn ended on "Does that work
    for you?" — a question about nothing. A legal notice is not a proposal
    and there is nothing in it for the consumer to assent to.

The scripted clients below are deliberately *not* ``MockLLMClient``: that
client asks ``_disclosed(messages)`` of the transcript before speaking the
notice, so it declines to repeat itself and can never exhibit the defect.
These clients reproduce what production actually did, which is the point —
the assertions are on what the deterministic layer permits to reach TTS, not
on what a well-behaved model chooses to say.
"""

from __future__ import annotations

from collector.agent import NegotiationAgent
from collector.guardrails.disclosures import (
    DisclosureId,
    DisclosureState,
    fires_mini_miranda,
)
from collector.guardrails.rings import PreCallContext
from collector.llm.base import LLMResponse, Message
from collector.policy import PolicyConfig

POLICY = PolicyConfig.default()

# The notice as the live model actually phrased it — not MINI_MIRANDA_TEXT,
# because what has to be caught is any complete delivery, not one canonical
# string. ``fires_mini_miranda`` is the arbiter of "complete".
NOTICE = (
    "This is an attempt to collect a debt by a debt collector and any "
    "information obtained will be used for that purpose."
)
OPENING = (
    "This is an AI calling on behalf of Meridian Recovery Services. "
    "A real person is available to speak with you if you prefer. "
    "Am I speaking with Dana Whitfield?"
)


class _RepeatsTheNotice:
    """Leads every post-identity turn with the notice, the way the live model
    did once its first attempt at the balance was stopped."""

    def __init__(self, tail: str) -> None:
        self.tail = tail
        self.calls = 0

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(text=OPENING)
        return LLMResponse(text=f"{NOTICE} {self.tail}")


def _opened(llm: object) -> NegotiationAgent:
    agent = NegotiationAgent(llm=llm, policy=POLICY)  # type: ignore[arg-type]
    agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
    return agent


class TestTheNoticeIsSpokenOnce:
    def test_the_voice_path_does_not_speak_a_second_mini_miranda(self) -> None:
        """The live-call reproduction, on the path that carries real calls.

        Turn one delivers the notice and latches it. Turn two must not carry
        it again, whatever the model writes: by then the disclosure is on
        record and the state machine knows it.
        """
        agent = _opened(
            _RepeatsTheNotice("There is an overdue balance of $8,400.00 on your account.")
        )

        first = " ".join(agent.stream_turn("Yes."))
        assert fires_mini_miranda(first) is not None, (
            f"the notice never reached the consumer on turn one: {first!r}"
        )
        assert DisclosureId.MINI_MIRANDA in agent.guard.disclosures.fired

        second = " ".join(agent.stream_turn("Yes."))
        assert fires_mini_miranda(second) is None, (
            "the Mini-Miranda was delivered a second time; it is already on "
            f"record and must not reach the consumer again: {second!r}"
        )

    def test_the_text_path_does_not_speak_a_second_mini_miranda(self) -> None:
        """Same rule on ``turn()``. The two paths share the outbound ring, so
        a control added to one and not the other is a control the voice path
        has and the text path does not — or the reverse.

        The tail here carries no figure, so nothing else can block the turn
        and the repeat is the only thing under test.
        """
        agent = _opened(_RepeatsTheNotice("What would work for you?"))

        first = agent.turn("Yes, this is her.")
        assert first.spoken is not None
        assert fires_mini_miranda(first.spoken) is not None, (
            f"the notice never reached the consumer on turn one: {first.spoken!r}"
        )

        second = agent.turn("Go on.")
        assert second.spoken is not None
        assert fires_mini_miranda(second.spoken) is None, (
            f"the Mini-Miranda was delivered a second time on the text path: {second.spoken!r}"
        )

    def test_a_repeat_never_reaches_the_transcript(self) -> None:
        """Not just the returned turn — the audit trail must not show the
        notice twice either. A call whose transcript carries two deliveries is
        a call that made two, however the turn object reads.
        """
        agent = _opened(_RepeatsTheNotice("What would work for you?"))
        agent.turn("Yes, this is her.")
        agent.turn("Go on.")
        agent.turn("Okay.")

        deliveries = [
            m.content
            for m in agent.messages
            if m.role == "agent" and fires_mini_miranda(m.content) is not None
        ]
        assert len(deliveries) == 1, (
            f"the notice appears {len(deliveries)} times in the transcript: {deliveries!r}"
        )


class TestTheStateMachineGatesTheRepeat:
    """Unit level. The latch works; what is missing is a rule that reads it.

    ``check_agent_turn`` is the only place the disclosure state gets a say
    over an outbound candidate, and today it is one-directional: it complains
    when the notice is *absent* and says nothing when it is redundant.
    """

    def test_check_agent_turn_flags_a_notice_already_on_record(self) -> None:
        state = DisclosureState.opening().observe_agent(
            "This is an AI calling from Meridian Recovery Services."
        )
        state = state.observe_agent(NOTICE)
        assert DisclosureId.MINI_MIRANDA in state.fired, "precondition: the notice has fired"

        violations = state.check_agent_turn(f"{NOTICE} There is a balance on the account.")

        assert any(v.blocking for v in violations), (
            "a candidate redelivering the Mini-Miranda after it has already "
            "fired must be blocked, not waved through; check_agent_turn "
            f"returned {violations!r}"
        )

    def test_a_turn_that_does_not_repeat_the_notice_is_untouched(self) -> None:
        """The complement, so the new rule cannot be satisfied by blocking
        everything once the notice is behind the call."""
        state = DisclosureState.opening().observe_agent(
            "This is an AI calling from Meridian Recovery Services."
        )
        state = state.observe_agent(NOTICE)

        assert state.check_agent_turn("There is a balance on the account.") == ()


class TestTheNoticeTurnIsNotWasted:
    def test_a_blocked_balance_does_not_close_the_notice_with_a_connective(self) -> None:
        """``_stands_on_its_own`` counts a completed Mini-Miranda as something
        a connective may close. It is not.

        The connective is "Does that work for you?" — a request for assent.
        Spoken over an offer the consumer just heard it closes a thought;
        spoken over a legal notice it asks them to agree to a disclosure,
        which is both meaningless and the exact shape the live call produced.
        A turn whose only surviving content is the notice has nothing to close
        and belongs on the scripted fallback, which at least asks a question
        the consumer can answer.
        """
        agent = _opened(
            _RepeatsTheNotice("There is an overdue balance of $8,400.00 on your account.")
        )

        spoken = " ".join(agent.stream_turn("Yes."))

        assert agent.turns[-1].blocked, "precondition: the balance sentence was blocked"
        assert fires_mini_miranda(spoken) is not None, "precondition: the notice was spoken"
        assert "Does that work for you?" not in spoken, (
            "the turn asked the consumer to assent to a legal notice; the "
            f"notice stands alone and closes nothing: {spoken!r}"
        )
