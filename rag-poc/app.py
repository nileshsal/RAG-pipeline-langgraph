# =============================================================================
# app.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the entry point of the whole project - the file you actually
# run to start the API server. It defines a single FastAPI endpoint,
# POST /ask, which:
#   1. Receives a JSON question from the user
#   2. Runs it through our LangGraph pipeline (retrieve -> generate)
#   3. Returns the final answer as JSON
#
# Run this file with:
#   uvicorn app:app --reload
# =============================================================================

from fastapi import FastAPI
from pydantic import BaseModel

from graph import rag_graph

# Create the FastAPI application. This "app" object is what uvicorn looks
# for when you run `uvicorn app:app`.
app = FastAPI(
    title="RAG POC - Pipeline 2 (Retrieval + Generation)",
    description="A minimal FastAPI + LangGraph service that answers "
    "questions using AWS OpenSearch retrieval and Claude generation.",
)


class AskRequest(BaseModel):
    """
    WHAT THIS IS:
    A Pydantic model describing the JSON body we expect on POST /ask.

    WHY WE NEED IT:
    FastAPI uses this to automatically validate incoming requests - if
    someone sends a request WITHOUT a "question" field, FastAPI will
    reject it with a clear error before our code ever runs.

    Example valid request body:
        { "question": "What is Generative AI?" }
    """

    question: str


class AskResponse(BaseModel):
    """
    WHAT THIS IS:
    A Pydantic model describing the JSON we send back.

    Example response body:
        { "answer": "Generative AI is..." }
    """

    answer: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    WHAT THIS ENDPOINT DOES:
    1. Takes the user's question from the request body.
    2. Builds the STARTING state for our LangGraph graph - at this point
       we only know the question; `retrieved_chunks` and `answer` are
       still empty and will be filled in as the graph runs.
    3. Calls rag_graph.invoke(...), which runs:
           START -> retrieve -> generate -> END
       and returns the FINAL state, with `retrieved_chunks` and `answer`
       now filled in.
    4. Returns just the `answer` field as the JSON response.
    """
    initial_state = {
        "question": request.question,
        "retrieved_chunks": [],
        "answer": "",
    }

    # .invoke() runs the entire graph from START to END and gives us
    # back the final state once every node has finished.
    final_state = rag_graph.invoke(initial_state)

    return AskResponse(answer=final_state["answer"])
