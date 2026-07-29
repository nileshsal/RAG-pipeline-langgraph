# =============================================================================
# app/models/schemas.py
# =============================================================================
# WHY THIS FILE EXISTS:
# These are the Pydantic models describing the JSON shapes our API
# accepts and returns. Keeping them separate from the route logic
# (app/api/routes/chat.py) means the "what does a request/response look
# like" question has one clear home.
# =============================================================================

from pydantic import BaseModel


class AskRequest(BaseModel):
    """
    WHAT THIS IS:
    A Pydantic model describing the JSON body we expect on
    POST /api/v1/chat/query.

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
