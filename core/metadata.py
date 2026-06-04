import ollama
import json

class MetadataEngine:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.system_prompt = (
            "You are a document classification and metadata extraction engine for an Indian SMB document store. "
            "Given raw document text, output ONLY a valid JSON object. No conversational filler.\n\n"
            "Fields:\n"
            "- collection: (legal, financial, gst, hr, compliance, operational)\n"
            "- doc_type: (e.g. tax_invoice, employment_agreement, balance_sheet, purchase_order, partnership_deed)\n"
            "- language: (hindi, english, hinglish, mixed)\n"
            "- entities: { 'pan': [], 'gstin': [], 'companies': [], 'people': [] }\n"
            "- date: (YYYY-MM-DD if found)\n"
            "- financial_year: (e.g. 2024-25)\n"
            "- summary: (one sentence description)"
        )

    def extract_metadata(self, text):
        """Extracts structured metadata from raw text using the LLM."""
        try:
            # We only send a sample of the text to keep it fast and within context limits
            # Metadata is usually found in the first 2000 characters
            sample_text = text[:2000]
            
            prompt = f"Document text:\n{sample_text}\n\nOutput JSON:"
            
            response = ollama.generate(
                model=self.model,
                system=self.system_prompt,
                prompt=prompt,
                format="json", # Ensure valid JSON output
                options={
                    "temperature": 0.1,
                    "num_ctx": 4096
                }
            )
            
            return json.loads(response['response'])
        except Exception as e:
            print(f"Metadata extraction error: {e}")
            return {
                "collection": "unknown",
                "doc_type": "unknown",
                "language": "unknown",
                "entities": {"pan": [], "gstin": [], "companies": [], "people": []},
                "date": None,
                "financial_year": None,
                "summary": "Could not extract summary."
            }

_metadata_engine_instance = None

def get_metadata_engine():
    global _metadata_engine_instance
    if _metadata_engine_instance is None:
        _metadata_engine_instance = MetadataEngine()
    return _metadata_engine_instance
