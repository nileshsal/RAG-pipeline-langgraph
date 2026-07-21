# =============================================================================
# opensearch_client.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the ONLY file that talks to AWS OpenSearch and AWS Bedrock
# (Titan Embeddings). Keeping all "talk to AWS" code in one file means:
#   - graph.py / nodes.py don't need to know HOW OpenSearch works
#   - if you ever need to change the search logic, there's only one place
#     to look
#
# What this file does, in plain English:
#   1. Turns the user's question into a vector (a list of numbers) using
#      the same Titan Embeddings model that was used to embed your
#      documents in Pipeline 1.
#   2. Sends that vector to OpenSearch and asks: "give me the top 3
#      chunks whose vectors are closest to this one".
#   3. Returns just the plain text of those chunks.
# =============================================================================

import json

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

import config


def get_opensearch_client() -> OpenSearch:
    """
    WHAT THIS FUNCTION DOES:
    Creates and returns a connected OpenSearch client, authenticated using
    your AWS credentials (the same credentials boto3 already knows about,
    e.g. from environment variables, an AWS profile, or an IAM role).

    WHY WE NEED IT:
    AWS OpenSearch Service requires requests to be "SigV4 signed" - a way
    of proving the request really comes from someone with valid AWS
    permissions. AWSV4SignerAuth handles that signing for us.
    """
    # boto3.Session().get_credentials() grabs whatever AWS credentials are
    # already configured on this machine (env vars, ~/.aws/credentials, or
    # an IAM role if running on AWS).
    credentials = boto3.Session().get_credentials()

    # "es" here means "Amazon OpenSearch Service" - it's the AWS service
    # name used for signing requests to it (a historical name from when
    # the service was called "Elasticsearch Service").
    auth = AWSV4SignerAuth(credentials, config.AWS_REGION, "es")

    client = OpenSearch(
        hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    return client


def get_query_embedding(text: str) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Sends `text` (the user's question) to Amazon Titan Embeddings (via
    AWS Bedrock) and returns the embedding - a list of numbers that
    represents the MEANING of that text.

    WHY WE NEED IT:
    OpenSearch can only compare vectors to vectors. Our documents are
    already stored as vectors (from Pipeline 1). So before we can search,
    we must convert the user's plain-text question into a vector using
    the SAME embedding model, so the two vectors "speak the same language".
    """
    # bedrock-runtime is the AWS client used to actually RUN a model
    # (as opposed to "bedrock", which is used to manage/list models).
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)

    # Titan Embeddings expects a JSON body shaped like: {"inputText": "..."}
    request_body = json.dumps({"inputText": text})

    response = bedrock_runtime.invoke_model(
        modelId=config.TITAN_EMBEDDING_MODEL_ID,
        body=request_body,
        contentType="application/json",
        accept="application/json",
    )

    # The response body is a stream of bytes containing JSON - we read it
    # and parse it to get the actual embedding list out.
    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]

    return embedding


def search_top_chunks(query_embedding: list, top_k: int = None) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Sends a "k-NN" (k-Nearest-Neighbors) search to OpenSearch: given a
    query vector, find the `top_k` stored documents whose vectors are
    closest (most similar in meaning) to it.

    WHY WE NEED IT:
    This is the actual "retrieval" step of Retrieval-Augmented Generation
    (RAG) - it's how we find the chunks of your original documents that
    are most relevant to the user's question.

    KEEPING IT SIMPLE (on purpose, per the project requirements):
    - No filters
    - No hybrid search (text search + vector search combined)
    - No re-ranking
    Just a plain k-NN vector search for the top_k closest chunks.
    """
    if top_k is None:
        top_k = config.TOP_K

    client = get_opensearch_client()

    # This is the simplest possible OpenSearch k-NN query shape:
    # "search the vector field for the k closest vectors to mine."
    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                config.VECTOR_FIELD_NAME: {
                    "vector": query_embedding,
                    "k": top_k,
                }
            }
        },
    }

    response = client.search(index=config.OPENSEARCH_INDEX, body=search_query)

    # response["hits"]["hits"] is a list of matching documents.
    # Each one has a "_source" dict containing the original fields we
    # stored in Pipeline 1 - we just want the text field out of each.
    hits = response["hits"]["hits"]
    chunks = [hit["_source"].get(config.TEXT_FIELD_NAME, "") for hit in hits]

    return chunks
