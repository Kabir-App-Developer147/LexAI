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

# Custom Prompt Template to prevent "mixing" and ensure source attribution
QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query. Please follow these rules:\n"
    "1. Be concise and practical.\n"
    "2. Always mention which file(s) you are referencing in your answer (e.g., 'According to [filename]...').\n"
    "3. If the answer is not in the context, say you don't know - don't try to make it up.\n"
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
        print("Loading existing index from ChromaDB...")
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        
        # Incremental Sync: Check for new files
        print("Checking for new or updated documents...")
        documents = ingestor.load_documents()
        # refresh_ref_docs detects changes based on doc_id (which is file_path by default)
        refreshed_docs = index.refresh_ref_docs(documents)
        if any(refreshed_docs):
            print(f"Refreshed {sum(refreshed_docs)} documents.")
        else:
            print("Index is already up to date.")
    else:
        print("No existing index found. Ingesting documents...")
        documents = ingestor.load_documents()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(f"Ingested {len(documents)} documents.")

    # Create query engine with custom prompt
    query_engine = index.as_query_engine(
        text_qa_template=QA_PROMPT,
        similarity_top_k=3
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
