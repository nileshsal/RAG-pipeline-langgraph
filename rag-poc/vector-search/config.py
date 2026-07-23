import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------
# AWS settings
# -----------------------------------------------------------------------
# The AWS region your OpenSearch domain and Bedrock (Titan Embeddings)
# live in. Example: "us-east-1"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# -----------------------------------------------------------------------
# AWS OpenSearch settings
# -----------------------------------------------------------------------
# The OpenSearch domain endpoint, WITHOUT "https://" in front.
# Example: "search-my-domain-abc123.us-east-1.es.amazonaws.com"
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")

# OpenSearch over HTTPS almost always uses port 443.
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "443"))

# The name of the index where Pipeline 1 stored your chunks + embeddings.
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX")


OPENSEARCH_SERVICE =os.getenv("OPENSEARCH_SERVICE", "aoss")
# The field name INSIDE each OpenSearch document that holds the vector
# (the embedding). This must match whatever field name Pipeline 1 used
# when it stored the embeddings.
VECTOR_FIELD_NAME = os.getenv("VECTOR_FIELD_NAME", "embedding")

# The field name INSIDE each OpenSearch document that holds the original
# chunk text. This must also match what Pipeline 1 used.
TEXT_FIELD_NAME = os.getenv("TEXT_FIELD_NAME", "content")
#TEXT_FIELD_NAME = os.getenv("TEXT_FIELD_NAME", "text")




# -----------------------------------------------------------------------
# Amazon Titan Embeddings settings
# -----------------------------------------------------------------------
# We call this SAME embedding model to turn the user's QUESTION into a
# vector, so it can be compared against the vectors already stored in
# OpenSearch. If Pipeline 1 used a different Titan model, change this to
# match it exactly - otherwise the vectors won't be comparable.
TITAN_EMBEDDING_MODEL_ID = os.getenv(
    "TITAN_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)

# -----------------------------------------------------------------------
# Anthropic Claude settings
# -----------------------------------------------------------------------
# Your Anthropic API key. Get one from https://console.anthropic.com/
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# The Claude model we ask to generate the final answer.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "anthropic.claude-sonnet-4-6")

# -----------------------------------------------------------------------
# Retrieval settings
# -----------------------------------------------------------------------
# How many chunks to fetch from OpenSearch for every question.
TOP_K = int(os.getenv("TOP_K", "3"))
