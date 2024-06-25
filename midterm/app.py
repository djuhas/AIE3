import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# function to log messages to the chainlit interface
async def log_message(message):
    await cl.Message(content=message).send()

# initialize the Qdrant client and verify the connection
async def initialize_qdrant_client():
    try:
        client = QdrantClient(url=QDRANT_API_URL, api_key=QDRANT_API_KEY)
        client.get_collections()  # this will throw an exception if the connection fails
        await log_message("Successfully connected to Qdrant.")
    except Exception as e:
        await log_message(f"Failed to connect to Qdrant: {e}")
    return client

# load and split documents into smaller chunks
async def load_and_split_documents():
    await log_message("Loading and splitting documents...")
    document_loader = PyPDFLoader("/app/data/airbnb.pdf")  # ensure the path is accessible within the container
    documents = document_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=30)
    split_documents = text_splitter.split_documents(documents)
    
    for i, doc in enumerate(split_documents):
        doc.metadata['document_id'] = f'doc_{i}'
    
    await log_message("Documents loaded and split successfully.")
    return split_documents

# delete all existing collections in Qdrant
async def delete_all_collections(client):
    await log_message("Deleting all existing collections...")
    collections = client.get_collections().collections
    for collection in collections:
        client.delete_collection(collection.name)
    await log_message("Deleted all existing collections.")

# create a new Qdrant collection
async def create_qdrant_collection(client):
    await log_message("Creating Qdrant collection...")
    embedding_dimension = 1536
    try:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
        )
        await log_message(f"Created collection: {QDRANT_COLLECTION}")
    except Exception as e:
        await log_message(f"Error creating collection: {e}")

# index the document chunks into the Qdrant collection
async def index_documents(client, split_documents, embedding_model):
    await log_message("Indexing documents...")
    qdrant = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embedding_model)
    qdrant.add_documents(split_documents)
    await log_message(f"Indexed {len(split_documents)} documents.")
    return qdrant

# combine the steps to delete old collections, create a new one, and index documents
async def create_or_load_qdrant_collection(client, split_documents, embedding_model):
    await delete_all_collections(client)
    await create_qdrant_collection(client)
    vectorstore = await index_documents(client, split_documents, embedding_model)
    return vectorstore

# initialize everything: Qdrant client, document loading, splitting, embedding, and collection creation
async def initialize_system():
    client = await initialize_qdrant_client()
    split_documents = await load_and_split_documents()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = await create_or_load_qdrant_collection(client, split_documents, embedding_model)
    return vectorstore

# run this when the chat starts to initialize everything and inform the user
@cl.on_chat_start
async def start_chat():
    await log_message("Initializing system, please wait...")
    vectorstore = await initialize_system()
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.1, "k": 5}
    )

    rag_prompt = PromptTemplate.from_template("""\
system
You are a professional auditor. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know. Ensure that your answer is unbiased and avoids relying on stereotypes.

user
User Query:
{query}

Context:
{context}

assistant
Your task is to provide a concise and clear answer based on the context provided and nothing else.
""")

    openai_chat_model = ChatOpenAI(model="gpt-4o", streaming=True)

    rag_chain = (
        {"context": itemgetter("query") | retriever, "query": itemgetter("query")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
    )
    cl.user_session.set("rag_chain", rag_chain)
    await log_message("System initialized. Ready to answer your questions.")

# this function processes user messages and provides answers
@cl.on_message
async def main(message: cl.Message):
    await log_message("Processing your query...")
    rag_chain = cl.user_session.get("rag_chain")
    msg = cl.Message(content="")
    response = await rag_chain.ainvoke(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    )

    context = response["context"]
    response_content = response["response"].content

    # indicate if the answer is from the provided document
    source_info = "\n\nSource: Provided document." if context else ""
    final_response = response_content + source_info

    # stream the final response content back to the user in real-time
    await msg.stream_token(final_response)
    await msg.send()

# running the chainlit app to start the application
if __name__ == "__main__":
    cl.run()
