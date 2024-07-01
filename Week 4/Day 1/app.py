import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig

# load environment variables from .env file
load_dotenv()

# get endpoints and token from environment variables
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

# load documents from text file
text_loader = TextLoader("./data/paul_graham_essays.txt")
documents = text_loader.load()

# split documents into chunks of 1000 characters, with 30 characters overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

# create embeddings using the Hugging Face API
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)

# check if vectorstore already exists
if os.path.exists("./data/vectorstore/index.faiss"):
    # load existing vectorstore
    vectorstore = FAISS.load_local(
        "./data/vectorstore", 
        hf_embeddings, 
        allow_dangerous_deserialization=True
    )
    hf_retriever = vectorstore.as_retriever()
    print("loaded vectorstore")
else:
    print("indexing files")
    os.makedirs("./data/vectorstore", exist_ok=True)

    # create new vectorstore and add documents in batches of 32
    for i in range(0, len(split_documents), 32):
        if i == 0:
            vectorstore = FAISS.from_documents(split_documents[i:i+32], hf_embeddings)
            continue
        vectorstore.add_documents(split_documents[i:i+32])

hf_retriever = vectorstore.as_retriever()

# define a template for the prompt
RAG_PROMPT_TEMPLATE = """\
system
you are a helpful assistant. you answer user questions based on provided context. 
if you can't answer the question with the provided context, say you don't know.

user
user query:
{query}

context:
{context}

assistant
"""

# create a prompt template from the string
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# create a hugging face endpoint for the language model
hf_llm = HuggingFaceEndpoint(
    endpoint_url=f"{HF_LLM_ENDPOINT}",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.environ["HF_TOKEN"]
)

# function to rename the assistant in the chat
@cl.author_rename
def rename(original_author: str):
    rename_dict = {
        "assistant" : "paul graham essay bot"
    }
    return rename_dict.get(original_author, original_author)

# function to set up the chat session
@cl.on_chat_start
async def start_chat():
    lcel_rag_chain = {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}| rag_prompt | hf_llm
    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

# function to handle incoming messages
@cl.on_message  
async def main(message: cl.Message):
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")
    msg = cl.Message(content="")
    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    await msg.send()
