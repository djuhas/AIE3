import os
import json
import qdrant_client
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import Qdrant

# load environment variables from .env file
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# initialize qdrant client
client = qdrant_client.QdrantClient(url="http://localhost:6333", api_key=QDRANT_API_KEY)

# delete the collection if it already exists
existing_collections = client.get_collections().collections
if any(collection.name == QDRANT_COLLECTION for collection in existing_collections):
    client.delete_collection(QDRANT_COLLECTION)
    print(f"Deleted existing collection: {QDRANT_COLLECTION}")

# create qdrant collection
embedding_dimension = 1536
client.create_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
)

# load and split documents
document_loader = PyPDFLoader("./data/airbnb.pdf")
documents = document_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

# add unique IDs to each document chunk
for i, doc in enumerate(split_documents):
    doc.metadata['document_id'] = f'doc_{i}'

# initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# embed all document chunks
embeddings = embedding_model.embed_documents([doc.page_content for doc in split_documents])

# save vectors and metadata to a file
vectors = []
for doc, vector in zip(split_documents, embeddings):
    vectors.append({
        'id': doc.metadata['document_id'],
        'vector': vector,
        'metadata': doc.metadata
    })

with open('qdrant_data.json', 'w') as f:
    json.dump(vectors, f)

print("Qdrant vectors and metadata saved to qdrant_data.json")
