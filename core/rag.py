from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def load_rag_engine(data_folder="data"):
    Settings.llm = Ollama(model="gemma3:4b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    documents = SimpleDirectoryReader(data_folder).load_data()
    index = VectorStoreIndex.from_documents(documents)
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