import os
import glob
from pinecone import (
    Pinecone,
    CloudProvider,
    AwsRegion,
    EmbedModel,
    ServerlessSpec
)
from typing import List
from google import genai
from google.genai import types
import time


# Constants
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "sources"

# Initialize Gemini and Pinecone clients
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(PINECONE_API_KEY)

def load_documents():
    '''Load all text documents from the sources directory.'''
    
    documents = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "sources/*.txt")
    files = glob.glob(path)
    
    for file_path in files:
        with open(file_path, 'r', encoding="utf8") as file:
            content = file.read()
            documents.append({"content": content, "metadata": {"source": file_path}})
    
    print(f"Found {len(documents)} sources")
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    '''Split documents into smaller chunks for better processing.'''
    
    chunks = []
    
    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        
        # Simple text splitting
        for i in range(0, len(content), chunk_size - chunk_overlap):
            if i > 0:
                start = i - chunk_overlap
            else:
                start = 0
                
            chunk_content = content[start:start + chunk_size]
            if chunk_content:
                chunks.append({"content": chunk_content, "metadata": metadata})
    
    return chunks


def get_embeddings(texts: List[str]):
    '''Generate embeddings for a list of texts using Gemini.'''

    index = pc.Index(INDEX_NAME)

    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return [embedding.values for embedding in response.embeddings]

def embed_documents(chunks, namespace):
    '''Embed documents and store them in Pinecone.'''

    # Get Pinecone index
    index = pc.Index(INDEX_NAME)
    
    # Prepare batches (Pinecone usually works well with batches of ~100)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i+batch_size]
        
        # Get text from each chunk
        texts = [chunk["content"] for chunk in chunk_batch]
        
        # Get embeddings
        embeddings = get_embeddings(texts)
        
        # Prepare data for Pinecone
        vectors = []
        for j, embedding in enumerate(embeddings):
            vectors.append({
                "id": f"chunk_{i+j}",
                "values": embedding,
                "metadata": chunk_batch[j]["metadata"]
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors, namespace=namespace)
        time.sleep(60) #here to prevent exceeding 100 embedding requests per minute limit for free keys

def search_documents(query, namespace, top_k=5):
    '''Search the vector store with the user query.'''

    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    
    # Search Pinecone
    index = pc.Index(INDEX_NAME)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        include_values=False
    )
    
    # Get matched documents
    docs_with_scores = []
    for match in results["matches"]:
        # Load the document content based on the source file
        with open(match["metadata"]["source"], 'r', encoding="utf-8") as f:
            content = f.read()
        docs_with_scores.append((content, match["score"]))
    
    return docs_with_scores

def get_analysis(query, documents):
    """Ask OpenAI to perform the response"""

    # Join all documents into a single context string
    context = "\n\n".join([doc for doc, _ in documents])
    
    # Create messages for Gemini
    messages = [
        (
            "Provide an answer to the user's query about Berkshire Hathaway."
            "Various sources will be provided will be provided."
            "Use those documents to best answer the question."
        ),
        f"Documents: {context}",
        query
    ]

    #Call Gemini API
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=messages
    )
    
    return response.text

if __name__ == "__main__":
    if not pc.has_index(INDEX_NAME): #Create index if it doesn't exist
        pc.create_index(
            name=INDEX_NAME, 
            dimension=3072,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # Step 1: Load document embeddings into Pinecone - only run this the first time
    docs = load_documents()
    chunks = chunk_documents(docs)
    embed_documents(chunks, namespace="chunks")

    # Step 2: Write a query
    user_query = "?"

    # Step 3: Check Pinecone for similar chunks
    docs_and_scores = search_documents(query=user_query, namespace="chunks")
    for _, score in docs_and_scores:
        print(f"Score: {score}")
    
    # Step 4: Put docs into prompt and send to OpenAI
    response = get_analysis(user_query, docs_and_scores)
    print(response) 