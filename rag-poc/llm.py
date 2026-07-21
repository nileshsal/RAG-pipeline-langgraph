# =============================================================================
# llm.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the ONLY file that talks to Claude (Anthropic's LLM). It takes
# the user's question + the chunks we retrieved from OpenSearch, builds a
# simple prompt, sends it to Claude, and returns just the final answer
# text.
#
# Keeping this separate from opensearch_client.py means: if you ever want
# to change the prompt wording, or swap models, you only touch this file.
# =============================================================================

import anthropic

import config

# We create ONE Anthropic client when this file is first imported, and
# reuse it for every request. Creating a new client per-request would
# work too, but this is slightly more efficient.
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# This is the "system prompt" - instructions that tell Claude HOW to
# behave, before it even sees the user's question. Keeping Claude
# grounded in "only use the provided context" is what makes this a RAG
# system instead of Claude just answering from its own general knowledge.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer only using the provided context. "
    'If the answer is not found, say: "I don\'t have enough information."'
)


def build_user_prompt(question: str, chunks: list) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Combines the retrieved chunks and the user's question into a single
    block of text, in the exact "Context / Question" shape Claude expects
    based on our system prompt.

    WHY WE NEED IT:
    Claude doesn't automatically know which chunks we retrieved - we have
    to literally paste them into the message we send it. This function
    just formats that message consistently.
    """
    if chunks:
        # We separate chunks with a blank line so Claude can tell where
        # one chunk ends and the next begins.
        context_text = "\n\n".join(chunks)
    else:
        # If OpenSearch found nothing, we still send a valid prompt -
        # Claude will see there's no context and (per the system prompt)
        # should say it doesn't have enough information.
        context_text = "No context was found for this question."

    return f"Context:\n{context_text}\n\nQuestion:\n{question}"


def ask_claude(question: str, chunks: list) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Sends the question + retrieved chunks to Claude and returns Claude's
    answer as a plain string.

    WHY WE NEED IT:
    This is the "generation" half of Retrieval-Augmented Generation (RAG).
    Retrieval (opensearch_client.py) found the relevant information;
    this function asks Claude to actually turn that information into a
    human-readable answer.
    """
    user_prompt = build_user_prompt(question, chunks)

    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    # response.content is a list of "content blocks". For a simple text
    # answer (no tools, no thinking), there is normally just one block of
    # type "text" - we look for it and return its text.
    for block in response.content:
        if block.type == "text":
            return block.text

    # This should not normally happen, but if Claude returns no text
    # block at all, we fail gracefully instead of crashing.
    return "I don't have enough information."
