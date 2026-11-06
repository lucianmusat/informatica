from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import conversation
import prompts

EXCERPT = ("manual.pdf", "Descale every 3 months.", 0.3)


def chat(*pairs):
    """[(question, answer), ...] -> message dicts."""
    out = []
    for q, a in pairs:
        out.append(conversation.user_message(q, []))
        out.append({"role": "assistant", "content": a})
    return out


def test_excerpts_go_into_the_turn_not_the_system_prompt():
    messages = [conversation.user_message("How often?", [EXCERPT])]
    payload = conversation.model_messages(messages)
    assert payload[0] == SystemMessage(content=prompts.SYSTEM_PROMPT)
    assert isinstance(payload[1], HumanMessage)
    assert '<excerpt id="1" source="manual.pdf">\nDescale every 3 months.\n</excerpt>' in payload[1].content
    assert payload[1].content.endswith("\n\nHow often?")


def test_next_prompt_continues_the_previous_one_exactly():
    messages = [conversation.user_message("How often?", [EXCERPT])]
    first = conversation.model_messages(messages)
    messages.append({"role": "assistant", "content": "Every 3 months.\n"})
    messages.append(conversation.user_message("And the filter?", []))
    second = conversation.model_messages(messages)
    assert second[: len(first)] == first
    assert second[2] == AIMessage(content="Every 3 months.\n")  # not stripped


def test_plain_question_has_no_excerpt_block():
    payload = conversation.model_messages(chat(("hi", "hello")))
    assert payload[1].content == "hi"


def test_summary_goes_after_the_fixed_system_prompt():
    payload = conversation.model_messages([], summary="They talked about coffee.")
    assert payload[0].content.startswith(prompts.SYSTEM_PROMPT)
    assert payload[0].content.endswith("They talked about coffee.")


def test_usage_from_final_chunk():
    usage = conversation.usage_from({
        "done": True, "prompt_eval_count": 320, "eval_count": 12,
        "prompt_eval_duration": 3_815_241_000, "eval_duration": 1_000_000_000,
    })
    assert usage == {"prompt_tokens": 320, "output_tokens": 12, "prompt_seconds": 3.82,
                     "output_seconds": 1.0, "load_seconds": 0.0}
    assert conversation.usage_from(None)["prompt_tokens"] == 0


# -- token-budget window ------------------------------------------------------

from chats import Chat  # noqa: E402


def long_chat(exchanges, chars=4000):
    """Each message ~chars characters, i.e. ~1000 tokens at 4 chars/token."""
    c = Chat(owner="bob@x.com")
    for i in range(exchanges):
        c.messages.append(conversation.user_message(f"q{i} " + "x" * chars, []))
        c.messages.append({"role": "assistant", "content": f"a{i} " + "y" * chars})
    return c


def measured(c, prompt_tokens, output_tokens=1000):
    """Pretend the last answer was produced from the current window."""
    payload = conversation.model_messages(c.messages[c.window_start:-1], c.window_summary)
    c.messages[-1]["usage"] = conversation.usage_from(
        {"prompt_eval_count": prompt_tokens, "eval_count": output_tokens}, c, payload)


ASK = conversation.user_message("next?", [])


def test_window_grows_until_the_budget():
    c = long_chat(5)
    measured(c, 9000)
    assert not conversation.fit_window(c, ASK, 12000, 6000)
    assert c.window_start == 0


def test_real_count_triggers_the_cut():
    c = long_chat(6)
    # ~48k characters would estimate ~13.7k tokens at the default ratio; the
    # real count is what decides.
    measured(c, 11500, output_tokens=600)
    assert conversation.prompt_tokens(c, ASK) > 12000
    assert conversation.fit_window(c, ASK, 12000, 6000, summary="earlier stuff")
    assert c.window_start % 2 == 0 and c.window_start > 0
    assert c.window_summary == "earlier stuff"
    payload = conversation.window_payload(c, ASK)
    assert payload[0].content.endswith("earlier stuff")
    ratio = conversation.chars_per_token(c.messages)
    assert conversation.estimate_tokens(conversation.payload_chars(payload), ratio) <= 6000


def test_cut_is_in_one_large_step_then_grows_again():
    c = long_chat(14)
    measured(c, 12500)
    conversation.fit_window(c, ASK, 12000, 6000)
    start = c.window_start
    # Next turn: answered from the new window, well under budget.
    c.messages += [ASK, {"role": "assistant", "content": "ok"}]
    measured(c, 6000, output_tokens=10)
    assert not conversation.fit_window(c, ASK, 12000, 6000)
    assert c.window_start == start


def test_window_summary_stays_frozen_until_the_next_cut():
    c = long_chat(3)
    c.window_start, c.window_summary = 2, "old summary"
    measured(c, 3000)
    assert not conversation.fit_window(c, ASK, 12000, 6000, summary="newer summary")
    assert c.window_summary == "old summary"


def test_moved_window_is_estimated_not_taken_from_stale_counts():
    c = long_chat(3)
    measured(c, 5000)  # measured on window 0
    assert conversation.prompt_tokens(c, ASK) > 6000
    c.window_start = 4  # e.g. reopened with a summary
    assert conversation.prompt_tokens(c, ASK) < 3000


def test_huge_last_exchange_is_kept_if_it_fits_the_budget():
    c = long_chat(3, chars=16000)  # ~4.6k tokens per message at 3.5 chars/token
    c.messages[-1]["usage"] = {}
    assert conversation.fit_window(c, ASK, 12000, 6000)
    assert c.window_start == 4  # last exchange (~9k) kept: over trim_to, under budget


def test_nothing_fits_drops_all_history():
    c = long_chat(2, chars=30000)
    assert conversation.fit_window(c, ASK, 12000, 6000)
    assert c.window_start == 4
    assert conversation.window_payload(c, ASK)[1:] == [conversation.model_message(ASK)]


def test_window_of_one_oversized_exchange_drops_it():
    c = long_chat(3, chars=30000)
    c.window_start = 4
    assert conversation.fit_window(c, ASK, 12000, 6000)
    assert c.window_start == 6
