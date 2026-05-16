from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def load_rag_engine(data_folder="data"):
    # Tell LlamaIndex to use Gemma locally instead of OpenAI
    Settings.llm = Ollama(model="gemma3:4b", request_timeout=120.0)
    
    # Use a small local embedding model — no internet needed after first download
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    
    # Load all documents from the data folder
    documents = SimpleDirectoryReader(data_folder).load_data()
    
    # Build the index — this is where the magic happens
    index = VectorStoreIndex.from_documents(documents)
    
    # Return a query engine
    return index.as_query_engine()


def ask(engine, question):
    response = engine.query(question)
    return str(response)


if __name__ == "__main__":
    print("Loading documents...")
    engine = load_rag_engine()
    print("Ready. Ask anything about your documents.\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        answer = ask(engine, question)
        print(f"\nLexAI: {answer}\n")