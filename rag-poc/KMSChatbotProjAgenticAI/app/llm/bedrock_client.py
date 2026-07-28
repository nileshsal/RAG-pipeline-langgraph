# =============================================================================
# app/llm/bedrock_client.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the ONLY file that talks to Claude (Anthropic's LLM). It takes
# the user's question + the chunks we retrieved from OpenSearch, builds
# a prompt, sends it to Claude, and returns just the final answer text.
#
# NOTE ON AUTHENTICATION:
# We call Claude through AMAZON BEDROCK's raw invoke_model() API - the
# exact same low-level pattern used for Titan Embeddings in
# vectordb/opensearch_client.py. This means there is NO separate
# ANTHROPIC_API_KEY anywhere in this project - Claude is authenticated
# using the SAME AWS credentials that vectordb/opensearch_client.py
# already uses for OpenSearch and Titan Embeddings.
# =============================================================================

import json

import boto3

# "as config" keeps every reference below (config.BEDROCK_REGION, etc.)
# identical to how this file worked before the folder reorganization.
from app.config import settings as config
from app.prompts.system_prompt import SYSTEM_PROMPT

# We create ONE Bedrock Runtime client when this file is first imported,
# and reuse it for every request.
#
# NOTE: config.BEDROCK_REGION, not TITAN_REGION or OPENSEARCH_REGION -
# Claude's model/inference-profile ARN lives in its own AWS region in
# this project, which may differ from where Titan Embeddings or
# OpenSearch live.
bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)


def build_user_prompt(question: str, chunks: list) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Combines the retrieved chunks and the user's question into a single
    block of text, in the exact "Context / Question" shape Claude
    expects based on our system prompt.
    """
    if chunks:
        context_text = "\n\n".join(chunks)
    else:
        context_text = "No context was found for this question."

    return f"Context:\n{context_text}\n\nQuestion:\n{question}"


def ask_claude(question: str, chunks: list) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Sends the question + retrieved chunks to Claude via Bedrock's
    invoke_model() API, and returns Claude's answer as a plain string.
    """
    user_prompt = build_user_prompt(question, chunks)

    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
    )

    response = bedrock_runtime.invoke_model(
        # config.ANTHROPIC_MODEL holds whatever your AWS account actually
        # needs here - this might be a short model name OR a full
        # "inference profile" ARN, depending on your account/region.
        modelId=config.ANTHROPIC_MODEL,
        body=request_body,
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())

    for block in response_body.get("content", []):
        if block.get("type") == "text":
            return block["text"]

    # This should not normally happen, but if Claude returns no text
    # block at all, we fail gracefully instead of crashing.
    return "I don't have enough information."
