# =============================================================================
# nodes.py
# =============================================================================
# WHY THIS FILE EXISTS:
# LangGraph works by moving a "State" (a shared piece of data) through a
# series of "Nodes" (plain Python functions). This file defines:
#   1. What that State looks like (GraphState)
#   2. The two Nodes that will process it (retrieve_node, generate_node)
#
# Think of State as a clipboard that gets passed from person to person:
#   - Person 1 (retrieve_node) writes "here are the relevant chunks" on it
#   - Person 2 (generate_node) reads the chunks off the clipboard, writes
#     "here's the final answer" on it
# Nobody needs to know how the OTHER person did their job - they just
# read what they need off the clipboard and add their own part.
# =============================================================================

from typing import List, TypedDict

import opensearch_client
import llm


class GraphState(TypedDict):
    """
    WHAT THIS IS:
    A TypedDict describing every piece of data that flows through our
    graph. TypedDict is just a normal Python dict, but with type hints
    so your editor (and LangGraph) know what keys to expect.

    WHY WE NEED IT:
    LangGraph needs to know the "shape" of the data it's passing between
    nodes. Every node receives a GraphState and returns a (partial)
    GraphState with updated fields.

    FIELDS:
    - question:         the user's original question (set before the
                         graph even starts running)
    - retrieved_chunks: the list of text chunks found by OpenSearch
                         (filled in by retrieve_node)
    - answer:            Claude's final answer
                         (filled in by generate_node)
    """

    question: str
    retrieved_chunks: List[str]
    answer: str


def retrieve_node(state: GraphState) -> dict:
    """
    NODE 1: RETRIEVE
    ------------------
    WHAT THIS NODE DOES:
    Reads `question` off the state, turns it into an embedding, searches
    OpenSearch for the top matching chunks, and writes those chunks back
    onto the state under `retrieved_chunks`.

    WHY WE'RE CALLING IT:
    This is the "R" in RAG (Retrieval-Augmented Generation) - before we
    ask Claude anything, we first go fetch the information Claude will
    need to answer accurately, instead of relying on Claude's own
    (possibly outdated or incomplete) built-in knowledge.

    NOTE ON THE RETURN VALUE:
    A LangGraph node doesn't need to return the WHOLE state - just a
    dict of the fields it wants to update. LangGraph merges this dict
    into the existing state automatically.
    """
    question = state["question"]

    # Step 1: turn the question into a vector using Titan Embeddings.
    query_embedding = opensearch_client.get_query_embedding(question)

    # Step 2: use that vector to find the most similar chunks in
    # OpenSearch.
    chunks = opensearch_client.search_top_chunks(query_embedding)

    # Step 3: hand the chunks back so LangGraph can update the state.
    return {"retrieved_chunks": chunks}


def generate_node(state: GraphState) -> dict:
    """
    NODE 2: GENERATE
    ------------------
    WHAT THIS NODE DOES:
    Reads `question` AND `retrieved_chunks` off the state, sends both to
    Claude, and writes Claude's answer back onto the state under
    `answer`.

    WHY WE'RE CALLING IT:
    This is the "G" in RAG (Retrieval-Augmented Generation) - now that we
    have the relevant chunks, we ask Claude to actually read them and
    write a helpful, grounded answer for the user.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    answer = llm.ask_claude(question, chunks)

    return {"answer": answer}
