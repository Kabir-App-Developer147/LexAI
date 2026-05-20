import os
import fitz
import docx2txt
from llama_index.core import Document

class DocumentIngestor:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def load_documents(self):
        docs = []
        for filename in os.listdir(self.data_folder):
            filepath = os.path.join(self.data_folder, filename)
            
            try:
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
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
        return docs

ingestor = DocumentIngestor()
