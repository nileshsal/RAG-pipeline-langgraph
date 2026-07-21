# RAG POC — Pipeline 2 (Retrieval + Generation)

A minimal, beginner-friendly Proof of Concept that answers questions using
**Retrieval-Augmented Generation (RAG)**: it retrieves relevant chunks
from AWS OpenSearch and asks Claude to answer using only that retrieved
context.

This is **Pipeline 2** of a two-pipeline project:

- **Pipeline 1** (already built, not part of this repo): ingests
  documents, splits them into chunks, generates Amazon Titan
  Embeddings for each chunk, and stores them in an AWS OpenSearch
  vector index.
- **Pipeline 2** (this repo): takes a user's question, retrieves the
  most relevant chunks from that same OpenSearch index, and asks
  Claude Sonnet 4.6 to generate an answer grounded in those chunks.

This project is intentionally **simple**: no authentication, no
streaming, no caching, no Docker, no multi-agent orchestration. It is
meant for learning and demoing, not production use.

---

## 1. Project Overview

```
User Question
     |
     v
FastAPI endpoint (POST /ask)
     |
     v
LangGraph starts execution
     |
     v
Node 1: Retrieve top-3 relevant chunks from AWS OpenSearch
     |
     v
Node 2: Send question + chunks to Claude Sonnet 4.6
     |
     v
Claude generates the final answer
     |
     v
Return { "answer": "..." } as JSON
```

---

## 2. Folder Structure

```
rag-poc/
├── app.py                 # FastAPI app + the /ask endpoint
├── graph.py                # Builds the LangGraph graph (retrieve -> generate)
├── nodes.py                 # Defines the State + the two node functions
├── opensearch_client.py     # All AWS OpenSearch + Titan Embeddings calls
├── llm.py                    # All calls to Claude (prompt building + API call)
├── config.py                 # Loads all settings from environment variables
├── requirements.txt          # Python dependencies
├── .env.example               # Template for your local .env file
└── README.md                  # This file
```

---

## 3. Installation

**Prerequisites:** Python 3.10+, an AWS account with access to OpenSearch
and Bedrock (Titan Embeddings), and an Anthropic API key.

```bash
# 1. Move into the project folder
cd rag-poc

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file from the template
cp .env.example .env
# then open .env and fill in your real values
```

---

## 4. Required Python Packages

Installed automatically via `requirements.txt`:

| Package | Purpose |
|---|---|
| `fastapi` | Web framework that exposes the `/ask` endpoint |
| `uvicorn[standard]` | Server that actually runs the FastAPI app |
| `langgraph` | Orchestrates the retrieve -> generate flow as a graph |
| `anthropic` | Official SDK for calling Claude |
| `boto3` | AWS SDK — used to call Bedrock (Titan Embeddings) |
| `opensearch-py` | Official client for querying AWS OpenSearch |
| `python-dotenv` | Loads `.env` file contents into environment variables |

---

## 5. Environment Variables

Set these in a `.env` file in the project root (see `.env.example`):

| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region your OpenSearch domain and Bedrock live in, e.g. `us-east-1` |
| `OPENSEARCH_HOST` | OpenSearch domain endpoint, without `https://` |
| `OPENSEARCH_PORT` | Usually `443` |
| `OPENSEARCH_INDEX` | Name of the index Pipeline 1 wrote chunks/embeddings into |
| `VECTOR_FIELD_NAME` | Field name holding the embedding vector in each document |
| `TEXT_FIELD_NAME` | Field name holding the original chunk text in each document |
| `TITAN_EMBEDDING_MODEL_ID` | Titan Embeddings model ID — must match what Pipeline 1 used |
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `ANTHROPIC_MODEL` | Claude model to use (`claude-sonnet-4-6`) |
| `TOP_K` | How many chunks to retrieve per question (default `3`) |

**AWS credentials:** this app authenticates to AWS OpenSearch and
Bedrock using boto3's standard credential lookup (environment
variables, an AWS CLI profile, or an IAM role) — there are no AWS
credentials in `.env`. If `aws configure` already works on your
machine, this app will use those same credentials.

---

## 6. How to Start the FastAPI Server

```bash
uvicorn app:app --reload
```

- `app:app` means "in the file `app.py`, use the object named `app`".
- `--reload` restarts the server automatically whenever you edit a file
  (handy while learning/demoing — remove it for anything long-running).

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Interactive API docs are automatically available at:
`http://127.0.0.1:8000/docs`

---

## 7. How to Call the API (curl)

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Generative AI?"}'
```

### Expected Output

```json
{
  "answer": "Generative AI refers to artificial intelligence systems that can create new content..."
}
```

If OpenSearch has no relevant chunks for the question, Claude is
instructed (via the system prompt) to respond with:

```json
{
  "answer": "I don't have enough information."
}
```

---

## 8. Complete Execution Flow (Request to Response)

1. **You send a request**: `POST /ask` with `{"question": "..."}`.
2. **FastAPI validates the request** using the `AskRequest` Pydantic
   model in `app.py` — a missing `question` field is rejected
   automatically.
3. **`app.py` builds the starting state** for LangGraph:
   `{"question": "...", "retrieved_chunks": [], "answer": ""}`.
4. **LangGraph starts running** (`rag_graph.invoke(...)`), beginning at
   `START`.
5. **Node 1 — `retrieve_node`** (in `nodes.py`) runs first:
   - Calls `opensearch_client.get_query_embedding(question)`, which
     sends the question to Amazon Titan Embeddings (via Bedrock) and
     gets back a vector.
   - Calls `opensearch_client.search_top_chunks(...)`, which sends a
     k-NN search to AWS OpenSearch and gets back the top 3 most
     similar chunks.
   - Writes those chunks into the state as `retrieved_chunks`.
6. **Node 2 — `generate_node`** (in `nodes.py`) runs next:
   - Calls `llm.ask_claude(question, chunks)`, which builds a prompt
     (system instructions + context + question) and sends it to
     Claude Sonnet 4.6 via the Anthropic API.
   - Writes Claude's reply into the state as `answer`.
7. **LangGraph reaches `END`** and returns the final state to `app.py`.
8. **`app.py` extracts `final_state["answer"]`** and returns it as
   `{"answer": "..."}` — the JSON response you receive back from curl.

---

## 9. Important Notes / Limitations (by design)

This is a POC, so several things are intentionally left out:

- No authentication on the `/ask` endpoint
- No streaming responses
- No conversation memory (every question is independent)
- No hybrid search or re-ranking — just a plain top-3 vector search
- No retries, caching, or production monitoring
- `VECTOR_FIELD_NAME` / `TEXT_FIELD_NAME` / `TITAN_EMBEDDING_MODEL_ID`
  **must exactly match** whatever Pipeline 1 used when it built the
  OpenSearch index — a mismatch here is the most common source of
  "no results" or dimension errors.
