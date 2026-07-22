import boto3
import json

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="ap-southeast-1"
)

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 500,
    "messages": [
        {
            "role": "user",
            "content": "Explain AWS Bedrock."
        }
    ]
}

response = bedrock.invoke_model(
    modelId="arn:aws:bedrock:ap-southeast-1:892880329963:inference-profile/global.anthropic.claude-sonnet-4-6",
    body=json.dumps(body),
    contentType="application/json"
)

print(json.loads(response["body"].read()))