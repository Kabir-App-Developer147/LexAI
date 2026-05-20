import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from core.ingest import ingestor
import os

# Configuration
DB_PATH = "./db/chroma_db"
COLLECTION_NAME = "lexai_docs"

def load_rag_engine():
    # Initialize Settings
    Settings.llm = Ollama(model="gemma3:4b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Check if index exists by checking if collection has items
    if chroma_collection.count() > 0:
        print("Loading existing index from ChromaDB...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        print("No existing index found. Ingesting documents...")
        documents = ingestor.load_documents()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(f"Ingested {len(documents)} documents.")

    return index.as_query_engine()

def ask(engine, question):
    response = engine.query(question)
    return str(response)

if __name__ == "__main__":
    print("Initializing LexAI RAG Engine...")
    engine = load_rag_engine()
    print("Ready. Ask anything about your documents.\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        answer = ask(engine, question)
        print(f"\nLexAI: {answer}\n")
