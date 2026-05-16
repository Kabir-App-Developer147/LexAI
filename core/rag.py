from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import docx2txt
import os
import fitz
from llama_index.core import Document

def load_documents(data_folder="data"):
    docs = []
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        
        if filename.endswith(".docx"):
            text = docx2txt.process(filepath)
            docs.append(Document(text=text, metadata={"filename": filename}))
            
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                docs.append(Document(text=f.read(), metadata={"filename": filename}))
                
        elif filename.endswith(".pdf"):
            pdf = fitz.open(filepath)
            text = ""
            for page in pdf:
                text += page.get_text()
            pdf.close()
            docs.append(Document(text=text, metadata={"filename": filename}))
            
    return docs

def load_rag_engine(data_folder="data"):
    Settings.llm = Ollama(model="gemma3:4b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    documents = load_documents(data_folder)
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