from dotenv import load_dotenv
import os
import json
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Initialize OpenAI with API key
openai.api_key = openai_api_key

# Define index name
index_name = "rag"

# Check if the index exists and delete it if necessary
existing_indexes = pc.list_indexes()

if index_name in existing_indexes:
    print(f"Index '{index_name}' already exists. Deleting it.")
    pc.delete_index(index_name)

# Create a new Pinecone index
try:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

# Load the review data
with open("reviews.json") as f:
    data = json.load(f)

processed_data = []

# Create embeddings for each review
for review in data["reviews"]:
    response = openai.Embedding.create(
        input=review['review'], model="text-embedding-3-small"
    )
    embedding = response['data'][0]['embedding']
    processed_data.append(
        {
            "values": embedding,
            "id": review["professor"],
            "metadata": {
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Initialize PineconeVectorStore
vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding)

# Insert the embeddings into the Pinecone index
index = pc.Index(index_name)
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())
