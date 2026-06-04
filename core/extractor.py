import ollama
import json

class ExtractorEngine:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.system_prompt = (
            "You are a structured data extraction engine for Indian business documents. "
            "Given raw text, extract all structured information into a single JSON object.\n\n"
            "Fields to include (set to null if missing):\n"
            "- doc_type, date, financial_year\n"
            "- parties: array of {name, address, gstin, pan, role}\n"
            "- line_items: array of {description, hsn_sac, quantity, unit, rate, taxable_amount, gst_rate, gst_amount, total}\n"
            "- totals: {taxable_value, total_gst, grand_total}\n"
            "- payment_terms, key_clauses, bank_details, signatures_required\n\n"
            "CRITICAL: Do NOT invent data. Output ONLY valid JSON."
        )

    def extract_structured_data(self, text):
        """Performs deep structured extraction from document text."""
        try:
            # Send enough text for deep extraction (first 4000 chars)
            sample_text = text[:4000]
            prompt = f"Raw document text: {sample_text}\n\nOutput JSON:"
            
            response = ollama.generate(
                model=self.model,
                system=self.system_prompt,
                prompt=prompt,
                format="json",
                options={"temperature": 0.1, "num_ctx": 8192}
            )
            return json.loads(response['response'])
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"error": str(e)}

_extractor_instance = None

def get_extractor():
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ExtractorEngine()
    return _extractor_instance
