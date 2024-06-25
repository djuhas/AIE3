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

# first, we load our environment variables from a .env file
# this is crucial because we don't want to hardcode sensitive information like API keys directly into our script
# it makes our code more secure and flexible since we can change these values without modifying the actual code
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# function to log messages to the chainlit interface
# this function will help us keep the user informed about what's happening behind the scenes
# for instance, when we're loading documents, connecting to Qdrant, or indexing data
async def log_message(message):
    await cl.Message(content=message).send()

# initialize the Qdrant client and verify the connection
# here, we're setting up a connection to the Qdrant database which will store our document vectors
# it's important to check if the connection is successful to avoid issues later when we try to store or retrieve data
async def initialize_qdrant_client():
    try:
        client = QdrantClient(url=QDRANT_API_URL, api_key=QDRANT_API_KEY)
        client.get_collections()  # this will throw an exception if the connection fails
        await log_message("Successfully connected to Qdrant.")
    except Exception as e:
        await log_message(f"Failed to connect to Qdrant: {e}")
    return client

# load and split documents into smaller chunks
# loading a large document as a single chunk isn't efficient for processing by the language model
# instead, we split it into smaller chunks which the model can handle better
# each chunk is given a unique ID so we can track which part of the document it came from
async def load_and_split_documents():
    await log_message("Loading and splitting documents...")
    document_loader = PyPDFLoader("./data/airbnb.pdf")  # specify the path to the PDF file
    documents = document_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=30)  # specify chunk size and overlap
    split_documents = text_splitter.split_documents(documents)
    
    # adding unique IDs to each document chunk
    # this is crucial for tracking and referencing each chunk later when we need to retrieve and use them
    for i, doc in enumerate(split_documents):
        doc.metadata['document_id'] = f'doc_{i}'
    
    await log_message("Documents loaded and split successfully.")
    return split_documents

# delete all existing collections in Qdrant
# this step is necessary to ensure we're starting fresh every time we run the script
# it prevents conflicts and ensures that old data doesn't interfere with our new data
async def delete_all_collections(client):
    await log_message("Deleting all existing collections...")
    collections = client.get_collections().collections
    for collection in collections:
        client.delete_collection(collection.name)
    await log_message("Deleted all existing collections.")

# create a new Qdrant collection
# a collection in Qdrant is like a table in a database where we will store our document vectors
# it's important to define the collection properly to match the dimensions of our embeddings
async def create_qdrant_collection(client):
    await log_message("Creating Qdrant collection...")
    embedding_dimension = 1536  # dimension of the embeddings, it must match the embedding model used
    try:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
        )
        await log_message(f"Created collection: {QDRANT_COLLECTION}")
    except Exception as e:
        await log_message(f"Error creating collection: {e}")

# index the document chunks into the Qdrant collection
# indexing involves adding the document chunks (with their embeddings) to the Qdrant collection
# this step is crucial because it allows us to perform efficient searches over our documents later
async def index_documents(client, split_documents, embedding_model):
    await log_message("Indexing documents...")
    qdrant = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embedding_model)
    qdrant.add_documents(split_documents)
    await log_message(f"Indexed {len(split_documents)} documents.")
    return qdrant

# combine the steps to delete old collections, create a new one, and index documents
# this function ensures we have a clean setup every time we initialize the system
async def create_or_load_qdrant_collection(client, split_documents, embedding_model):
    await delete_all_collections(client)
    await create_qdrant_collection(client)
    vectorstore = await index_documents(client, split_documents, embedding_model)
    return vectorstore

# initialize everything: Qdrant client, document loading, splitting, embedding, and collection creation
# this function orchestrates all the steps required to set up our system from scratch
async def initialize_system():
    client = await initialize_qdrant_client()
    split_documents = await load_and_split_documents()
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")  # use the specified OpenAI embedding model
    vectorstore = await create_or_load_qdrant_collection(client, split_documents, embedding_model)
    return vectorstore

# run this when the chat starts to initialize everything and inform the user
# it sets up the document retriever and the prompt template for generating responses
@cl.on_chat_start
async def start_chat():
    await log_message("Initializing system, please wait...")
    vectorstore = await initialize_system()
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",  # use similarity score threshold for searching
        search_kwargs={"score_threshold": 0.1, "k": 5}  # parameters for the search: score threshold and number of results
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

    openai_chat_model = ChatOpenAI(model="gpt-4", streaming=True)  # use the GPT-4 model for generating responses

    # combine the retriever and prompt template into a processing chain
    rag_chain = (
        {"context": itemgetter("query") | retriever, "query": itemgetter("query")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
    )
    cl.user_session.set("rag_chain", rag_chain)
    await log_message("System initialized. Ready to answer your questions.")

# this function processes user messages and provides answers
# it retrieves relevant documents using the retriever and generates responses based on them
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
    # this helps the user understand the source of the information
    source_info = "\n\nSource: Provided document." if context else ""
    final_response = response_content + source_info

    # stream the final response content back to the user in real-time
    await msg.stream_token(final_response)
    await msg.send()

# running the chainlit app to start the application
if __name__ == "__main__":
    cl.run()
