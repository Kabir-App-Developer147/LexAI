import os
from llama_index.core import SimpleDirectoryReader

class DocumentIngestor:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def load_documents(self):
        """
        Loads all documents from the data folder using SimpleDirectoryReader.
        This is significantly more robust for .pdf and .docx files than manual 
        parsing, as it correctly handles encoding, images, and structural 
        metadata (like page numbers and filenames).
        """
        if not os.path.exists(self.data_folder) or not os.listdir(self.data_folder):
            print(f"Warning: Data folder '{self.data_folder}' is empty.")
            return []
            
        # SimpleDirectoryReader automatically detects file types and extracts metadata
        reader = SimpleDirectoryReader(input_dir=self.data_folder)
        return reader.load_data()

_ingestor_instance = None

def get_ingestor(data_folder="data"):
    global _ingestor_instance
    if _ingestor_instance is None:
        _ingestor_instance = DocumentIngestor(data_folder=data_folder)
    return _ingestor_instance
