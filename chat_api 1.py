from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import ollama
from huggingface_hub import InferenceClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large") # alternative model

Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25


# Define FastAPI app
app = FastAPI()
token = "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"

# Available models and API URLs
model_clients = {
    "meta-llama": InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=token),
    "phi-3.5":  InferenceClient("microsoft/Phi-3.5-mini-instruct", token=token),
    "phi 3": InferenceClient("microsoft/Phi-3-mini-4k-instruct", token=token) ,
    "mistralai": InferenceClient("mistralai/Mistral-Nemo-Instruct-2407", token=token) ,
    "nousresearch": InferenceClient("NousResearch/Hermes-3-Llama-3.1-8B", token=token) ,
    "ollama": "local"
}

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory store for selected models per user (replace with session management for real apps)
user_model_choice = {}

user_rag_preference = {}

# Load index for retriever setup
storage_context = StorageContext.from_defaults(persist_dir="C:/Users/Dora Sperling/Desktop/data6")
index = load_index_from_storage(storage_context)
top_k = 3

retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
)

# Request and Response models for FastAPI
class ModelSelection(BaseModel):
    user_id: str
    name_model: str

# Request and Response models for FastAPI
class RagPreference(BaseModel):
    user_id: str
    prefers_rag: bool

class ChatRequest(BaseModel):
    user_id: str
    message: str
    model: str 
    use_rag: bool

class UserQuery(BaseModel):
    user_query: str

# Helper functions
def get_rag_context_documents(query):
    response = query_engine.query(query)
    if len(response.source_nodes) > 0:
        context = "Documents list:\n"
        number_of_documents = len(response.source_nodes)
        for i in range(number_of_documents):
            context = context + f"Document {i}: " + response.source_nodes[i].text + "\n\n"
        return context
    return ""

def prompt_template(comment, context):
    return f"""
    {context}

    Please respond to the following comment. If the comment is related to health or medicine, respond in the tone of an authoritative medical professional answering a patient or medical student. Keep the response brief (less than 100 words). Do not introduce yourself or write an introduction of any kind.

    {comment}
    """

def query_ollama(prompt):
    stream = ollama.chat(
        model='llama3.1:8b',
        stream=True,
        messages=[{'role': 'user', 'content': prompt}]
    )
    for chunk in stream:
        yield chunk['message']['content']

def query(client, payload):
    messages = [{"role": "user", "content": payload}]
    for token in client.chat_completion(messages, max_tokens=1000, stream=True, temperature=1):
        yield token.choices[0].delta.content



@app.post("/rag-documents")
async def get_rag_documents(query:UserQuery):
    user_query = query.user_query
    
    response = query_engine.query(user_query)
    rag_documents = []
    if len(response.source_nodes) > 0:
        number_of_documents = len(response.source_nodes)
        
        for i in range(number_of_documents):
            rag_doc = {}
            rag_doc['text'] = response.source_nodes[i].text
            rag_doc['filename'] = str(response.source_nodes[i].metadata['file_name'])
            rag_documents.append(rag_doc)
    return rag_documents
    



@app.post("/chat/")
async def chat(request: ChatRequest):
    user_id = request.user_id
    message = request.message
    model = request.model
    use_rag = request.use_rag

    use_rag = user_rag_preference.get(user_id, True)
    
    context=""
    
    if use_rag:
        print("using rag")    
        context = get_rag_context_documents(message)
        
    prompt = prompt_template(message, context)

    if model == "ollama":
        # Local Ollama model
        response_stream = query_ollama(prompt)
    else:
        # Hugging Face API model
        client = model_clients[model]
        response_stream = query(client, prompt)

    return StreamingResponse(response_stream, media_type="text/plain")
