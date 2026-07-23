import json

import boto3
from opensearchpy import OpenSearch, RequestsAWSV4SignerAuth, RequestsHttpConnection
from opensearchpy.exceptions import TransportError

import config


def get_opensearch_client() -> OpenSearch:

    # boto3.Session().get_credentials() grabs whatever AWS credentials are
    # already configured on this machine (env vars, ~/.aws/credentials, or
    # an IAM role if running on AWS).
    credentials = boto3.Session().get_credentials()

    # Print exactly which AWS identity (IAM user/role) is about to be
    # used - this is the identity that needs to be allow-listed in the
    # OpenSearch Serverless data access policy.
    try:
        identity = boto3.client("sts", region_name=config.AWS_REGION).get_caller_identity()
        print("[opensearch_client] Calling AWS as identity:")
        print(f"    Account: {identity.get('Account')}")
        print(f"    Arn:     {identity.get('Arn')}")
    except Exception as identity_error:  # noqa: BLE001 - diagnostic only
        print(f"[opensearch_client] Could NOT resolve AWS identity: {identity_error}")

    print("[opensearch_client] Connection settings being used:")
    print(f"    OPENSEARCH_HOST:    {config.OPENSEARCH_HOST}")
    print(f"    OPENSEARCH_PORT:    {config.OPENSEARCH_PORT}")
    print(f"    OPENSEARCH_INDEX:   {config.OPENSEARCH_INDEX}")
    print(f"    OPENSEARCH_SERVICE: {config.OPENSEARCH_SERVICE}")
    print(f"    AWS_REGION:         {config.AWS_REGION}")

    auth = RequestsAWSV4SignerAuth(
        credentials, "ap-southeast-1", config.OPENSEARCH_SERVICE
    )

    client = OpenSearch(
        hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client


def get_query_embedding(text: str) -> list:

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

    print(f"[opensearch_client] Searching index: {config.OPENSEARCH_INDEX!r}")
    print(
        f"[opensearch_client] Query shape: knn on field "
        f"{config.VECTOR_FIELD_NAME!r}, k={top_k}, vector length="
        f"{len(query_embedding)}"
    )

    # DIAGNOSTIC: we wrap this call so that if OpenSearch rejects it, we
    # print the FULL detail AWS actually sent back - not just "403
    # forbidden" - before letting the error continue up (re-raising),
    # so FastAPI still reports the failure like normal.
    try:
        response = client.search(index=config.OPENSEARCH_INDEX, body=search_query)
    except TransportError as search_error:
        print("[opensearch_client] SEARCH FAILED - full AWS error detail below:")
        print(f"    status_code: {search_error.status_code}")
        print(f"    error:       {search_error.error}")
        # .info is normally a dict with AWS's detailed reason - print it
        # fully formatted so it's easy to read and easy to copy/paste.
        try:
            print(f"    info:        {json.dumps(search_error.info, indent=4)}")
        except TypeError:
            # .info isn't always JSON-serializable (e.g. a raw exception) -
            # fall back to printing it as-is rather than crashing here.
            print(f"    info:        {search_error.info!r}")
        raise

    # response["hits"]["hits"] is a list of matching documents.
    # Each one has a "_source" dict containing the original fields we
    # stored in Pipeline 1 - we just want the text field out of each.
    hits = response["hits"]["hits"]
    chunks = [hit["_source"].get(config.TEXT_FIELD_NAME, "") for hit in hits]

    return chunks



def main():

    query = input("Enter your query: ")

    print("\nGenerating embedding...")

    query_embedding = get_query_embedding(query)

    print(f"Embedding generated. Length = {len(query_embedding)}")

    print("\nSearching OpenSearch...")

    results = search_top_chunks(
        query_embedding=query_embedding,
        top_k=5
    )

    print("\nTop Matches")
    print("=" * 80)

    # Debug
    print(f"\nResults Type: {type(results)}")

    if not results:
        print("No results found.")
        return

    for idx, item in enumerate(results, start=1):

        print("\n" + "=" * 80)
        print(f"Rank #{idx}")

        # Case 1: Raw text chunk returned
        if isinstance(item, str):

            print("Type : Text Chunk")
            print(f"Text : {item[:1000]}")

        # Case 2: OpenSearch hit returned
        elif isinstance(item, dict):

            score = item.get("_score", "N/A")
            source = item.get("_source", {})

            print(f"Score : {score}")
            print(
                f"Document : {source.get('document_name', 'NA')}"
            )
            print(
                f"Chunk Id : {source.get('chunk_id', 'NA')}"
            )
            print(
                f"Text : {source.get('text', '')[:1000]}"
            )

        # Case 3: Unknown object
        else:

            print(f"Unexpected result type: {type(item)}")
            print(item)


if __name__ == "__main__":
    main()