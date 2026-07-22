import json
import boto3

from opensearchpy import (
    OpenSearch,
    RequestsHttpConnection,
    RequestsAWSV4SignerAuth
)

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

HOST = "mc9xamsc1uiy9cxgnwt9.aoss.ap-southeast-1.on.aws"

REGION = "ap-southeast-1"

SERVICE = "aoss"

INDEX_NAME = "kms-index"

# ------------------------------------------------
# AWS SESSION
# ------------------------------------------------

print("=" * 80)
print("STEP-1 : AWS Identity")
print("=" * 80)

session = boto3.Session()

credentials = session.get_credentials()

identity = boto3.client("sts").get_caller_identity()

print(json.dumps(identity, indent=4))

# ------------------------------------------------
# AUTH
# ------------------------------------------------

print("\n" + "=" * 80)
print("STEP-2 : Creating Auth")
print("=" * 80)

auth = RequestsAWSV4SignerAuth(
    credentials,
    REGION,
    SERVICE
)

print("SUCCESS")

# ------------------------------------------------
# CLIENT
# ------------------------------------------------

print("\n" + "=" * 80)
print("STEP-3 : Creating Client")
print("=" * 80)

client = OpenSearch(
    hosts=[
        {
            "host": HOST,
            "port": 443
        }
    ],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True
)

print("SUCCESS")

# ------------------------------------------------
# CHECK INDEX
# ------------------------------------------------

print("\n" + "=" * 80)
print("STEP-4 : Check Index")
print("=" * 80)

exists = client.indices.exists(
    index=INDEX_NAME
)

print("Index Exists =", exists)

# ------------------------------------------------
# SEARCH EXISTING DOCS
# ------------------------------------------------

print("\n" + "=" * 80)
print("STEP-5 : Search")
print("=" * 80)

response = client.search(
    index=INDEX_NAME,
    body={
        "size": 10,
        "query": {
            "match_all": {}
        }
    }
)

print(json.dumps(response, indent=2))

# ------------------------------------------------
# INSERT DOCUMENT
# ------------------------------------------------

print("\n" + "=" * 80)
print("STEP-6 : Insert Document")
print("=" * 80)

try:

    response = client.index(
        index=INDEX_NAME,
        body={
            "message": "Hello OpenSearch",
            "source": "EC2",
            "application": "KMS Chatbot"
        }
    )

    print("INSERT SUCCESS")

    print(json.dumps(response, indent=2))

except Exception as e:

    print("INSERT FAILED")

    print(str(e))

# ------------------------------------------------
# SEARCH AGAIN
# ------------------------------------------------

print("\n" + "=" * 80)
print("STEP-7 : Verify")
print("=" * 80)

response = client.search(
    index=INDEX_NAME,
    body={
        "size": 20,
        "query": {
            "match_all": {}
        }
    }
)

print(json.dumps(response, indent=2))

print("\nDONE")