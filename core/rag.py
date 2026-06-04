import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from core.ingest import get_ingestor
import os

# Configuration
DB_PATH = "./db/chroma_db"
COLLECTION_NAME = "lexai_docs"

# Distilled LexAI RAG Prompt
QA_PROMPT_TMPL = (
    "# IDENTITY\n"
    "You are LexAI, the street-smart business assistant. Answer based ONLY on context.\n\n"
    "# CONTEXT\n"
    "{context_str}\n\n"
    "# RULES\n"
    "1. Citations: Start with 'As per [filename]...' or 'Checking [filename]...'.\n"
    "2. If unsure: Say 'Don't know, boss' with friendly local flair.\n"
    "3. Tone: Practical, concise, Indian business context.\n"
    "4. Playful: If user says 'oye oye', respond 'Oye oye cappin!' first.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

def load_rag_engine():
    # Initialize Settings
    Settings.llm = Ollama(model="gemma3:4b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    ingestor = get_ingestor()
    
    # Load or create index
    if chroma_collection.count() > 0:
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        # Sync logic...
        documents = ingestor.load_documents()
        index.refresh_ref_docs(documents)
    else:
        documents = ingestor.load_documents()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

    # TWO-STAGE RETRIEVAL: Fetch more chunks (8) to ensure high-quality re-ranking 
    # capability even without a secondary model.
    query_engine = index.as_query_engine(
        text_qa_template=QA_PROMPT,
        similarity_top_k=8 
    )
    return query_engine

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
