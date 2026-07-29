# =============================================================================
# test_json_normalizer.py
# =============================================================================
# WHY THIS FILE EXISTS:
# A small, standalone script to see EXACTLY what Claude's REAL raw reply
# looks like, and what utils/json_normalizer_dynamic.py's
# get_json_string() turns it into - printed straight to your terminal,
# without needing to run the full FastAPI server or use Postman.
#
# IMPORTANT: unlike a fully offline test, this script makes a REAL call
# to AWS Bedrock (via llm.py's get_raw_claude_text()), so it needs your
# real .env file (AWS credentials, ANTHROPIC_MODEL, etc.) to work - the
# same setup the main app already uses.
#
# HOW TO RUN IT:
#   python test_json_normalizer.py
#
# (Run it from the project root - the same folder that contains llm.py
# and the "utils" folder - so the imports below can find them.)
# =============================================================================

from llm import get_raw_claude_text
from utils.json_normalizer_dynamic import get_json_string


def main():
    """
    WHAT THIS FUNCTION DOES:
    1. Sends a test question (with a fake "chunk" standing in for real
       OpenSearch results) to Claude via get_raw_claude_text() - a REAL
       Bedrock call, same as the full app makes.
    2. Prints Claude's RAW reply exactly as Claude wrote it - before any
       cleanup - so you can see what get_json_string() is actually
       working with.
    3. Passes that raw text through get_json_string() and prints the
       cleaned-up result.
    """
    question = "How to create order"

    # A fake chunk standing in for real OpenSearch results - replace
    # this with real text if you want to test against something closer
    # to what your actual retrieval step would find.
    chunks = ["To create an order, open the Orders module and click New."]

    print("=" * 80)
    print("STEP 1: Calling Claude via Bedrock (get_raw_claude_text)...")
    print("=" * 80)
    raw_text = get_raw_claude_text(question, chunks)

    print()
    print("RAW TEXT (exactly what Claude sent back, before any cleanup):")
    print("-" * 80)
    print(raw_text)

    print()
    print("=" * 80)
    print("STEP 2: Passing that raw text through get_json_string...")
    print("=" * 80)
    # get_json_string() expects {"answer": <raw text>} - the same shape
    # llm.py's ask_claude() uses.
    filtered_json = get_json_string({"answer": raw_text})

    print()
    print("OUTPUT (what get_json_string returned):")
    print("-" * 80)
    print(filtered_json)


# This "if __name__ == ...:" guard means main() only runs when you
# execute this file directly (python test_json_normalizer.py) - not if
# this file ever gets imported from somewhere else.
if __name__ == "__main__":
    main()
