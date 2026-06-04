import os
from llama_index.core import SimpleDirectoryReader

from llama_index.core.node_parser import SentenceSplitter
from core.metadata import get_metadata_engine
from core.extractor import get_extractor

class DocumentIngestor:
    # ... (__init__ logic)

    def load_documents(self):
        """Loads and parses documents with metadata AND structured extraction."""
        all_docs = []
        meta_engine = get_metadata_engine()
        structured_engine = get_extractor()
        
        for folder in self.data_folders:
            if os.path.exists(folder) and os.listdir(folder):
                reader = SimpleDirectoryReader(
                    input_dir=folder, recursive=True, filename_as_id=True
                )
                docs = reader.load_data()
                
                for doc in docs:
                    print(f"Deep Analyzing: {doc.metadata.get('file_name', 'Unknown')}")
                    # 1. Classification
                    doc.metadata.update(meta_engine.extract_metadata(doc.text))
                    # 2. Structured Extraction (Parties, GST, Line Items)
                    doc.metadata['structured_data'] = structured_engine.extract_structured_data(doc.text)
                
                all_docs.extend(docs)
        return all_docs

_ingestor_instance = None

def get_ingestor(data_folders=None):
    global _ingestor_instance
    if _ingestor_instance is None:
        _ingestor_instance = DocumentIngestor(data_folders=data_folders)
    return _ingestor_instance
