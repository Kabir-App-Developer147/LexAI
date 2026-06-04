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

# Ultimate Master LexAI RAG Prompt
QA_PROMPT_TMPL = (
    "# IDENTITY & SILENT MEMORY\n"
    "You are a persistent business assistant (LexAI) with memory of this user's business.\n"
    "Do NOT repeat the profile back—use it SILENTLY to give more relevant, natural answers.\n"
    "Primary Language: English.\n\n"
    "# MISSION & EXPERTISE\n"
    "Answer based ONLY on context. If user gives new info, note it for profile updates.\n"
    "Specialized in GST (Section citations required), Indian Tax, and Legal drafting.\n\n"
    "# RULES & INTEGRITY\n"
    "1. READY-TO-USE: Produce complete drafts, not blanks.\n"
    "2. CITATIONS: Always cite specific Section Numbers for laws and document names.\n"
    "3. ZERO FABRICATION: Never make up GSTINs, PANs, or figures.\n"
    "4. LANGUAGE: Exactly mirror user language (Hindi, Hinglish, English).\n"
    "5. FALLBACK: If not in context, say: 'I could not find this in your documents, boss.'\n\n"
    "# CONTEXT\n"
    "{context_str}\n\n"
    "Query: {query_str}\n"
    "Answer: "
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

def load_rag_engine():
    # Initialize Settings
    Settings.llm = Ollama(
        model="gemma3:4b", 
        request_timeout=120.0,
        additional_kwargs={
            "num_ctx": 4096,
            "num_gpu": 1,
            "f16_kv": True
        }
    )
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    # SMALL-TO-BIG: Search tiny chunks, but give model context around them
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.node_parser = node_parser

    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    ingestor = get_ingestor() # Use defaults to get both data and db/kb

    # Load or create index
    if chroma_collection.count() > 0:
        print("Loading existing index from ChromaDB...")
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        # Sync logic
        print("Syncing documents from data and db/kb...")
        documents = ingestor.load_documents()
        refreshed = index.refresh_ref_docs(documents)
        print(f"Index sync complete. {sum(refreshed)} documents updated/added.")
    else:
        print("No existing index found. Ingesting documents from data and db/kb...")
        documents = ingestor.load_documents()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(f"Initial ingestion complete. {len(documents)} documents indexed.")


    # TWO-STAGE: Metadata replacement fetches the 'big' context for the 'small' search hit
    query_engine = index.as_query_engine(
        text_qa_template=QA_PROMPT,
        similarity_top_k=5,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ]
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
