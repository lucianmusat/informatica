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
