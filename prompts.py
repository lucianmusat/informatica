"""Prompts.

The assistant is a generalist first. Document excerpts are added only when the
retriever actually found something close, and even then the instructions tell the
model to fall back on its own knowledge rather than refusing — the old prompt's
"use ONLY the provided context" is what made this feel like a document-search box
instead of a chatbot.

The system prompt never changes between turns. Excerpts go into the user message
of the turn they were retrieved for and stay in the history, so each prompt is an
exact continuation of the previous one. llama.cpp can then reuse its cache and
read only the new turn; any change near the start of the prompt (as excerpts in
the system prompt used to cause) forces it to re-read the whole conversation.
"""

SYSTEM_PROMPT = """You are Informatica, a helpful and knowledgeable AI assistant.

Answer whatever the user brings you: coding, writing, analysis, general knowledge, \
or just conversation. Be direct and get to the point — skip filler openings like \
"Great question!". Match your length to the question: a sentence for simple things, \
more when the topic genuinely needs it.

Format with Markdown. Put code in fenced blocks tagged with the language. Use lists \
and headings when they aid scanning, not by reflex.

If you don't know something or aren't sure, say so plainly instead of inventing an \
answer.

The user has a personal document library. Files they attach to a message are \
added to it and named at the end of that message. Some user messages start with a \
<retrieved_excerpts> block: passages retrieved automatically as possibly relevant \
to that message. The user has not necessarily read them or asked about them. Use an \
excerpt only if it genuinely helps answer the question, and name the file it came \
from when you rely on it. If the excerpts are irrelevant, ignore them completely \
and answer normally from your own knowledge; do not mention that you were shown \
unhelpful excerpts, and do not apologise for them."""


SUMMARY_PREAMBLE = """Earlier parts of this conversation are no longer shown. \
This is a summary of them:"""


SUMMARY_REQUEST = """Pause the conversation and write a summary of it so far, \
including anything from the summary of earlier parts if there is one. It will \
replace the older messages, so include what is needed to continue the \
conversation: the topics discussed, decisions made, facts and figures the user \
gave or that were established, files referred to, and open questions. Write \
plain prose or short bullet points, at most about 300 words. Reply with the \
summary only."""


def system_prompt(summary: str = "") -> str:
    if not summary:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n---\n\n{SUMMARY_PREAMBLE}\n\n{summary}"


def user_content(question: str, excerpts, files=None) -> str:
    """What the model sees for one user turn: excerpts first, then the question,
    then the names of files attached to it (so "this file" means something).

    ``excerpts`` are (file_name, text, distance) triples, as stored in the chat.
    The output must be identical every time it is built for the same turn.
    """
    if files:
        question = f"{question}\n\n(Attached: {', '.join(files)})"
    if not excerpts:
        return question
    blocks = [
        f'<excerpt id="{i}" source="{file_name}">\n{text}\n</excerpt>'
        for i, (file_name, text, _distance) in enumerate(excerpts, start=1)
    ]
    return "<retrieved_excerpts>\n" + "\n\n".join(blocks) + "\n</retrieved_excerpts>\n\n" + question
