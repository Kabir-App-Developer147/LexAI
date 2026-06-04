import ollama
import json

class SyntheticDataGenerator:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.system_prompt = (
            "You are a high-quality training data generator for LexAI. "
            "Your task is to generate 10 realistic question-and-answer pairs for an Indian SMB owner dataset."
        )

    def generate_pairs(self, topic: str):
        """Generates 10 synthetic Q&A pairs for a given topic."""
        prompt = (
            f"Topic: {topic}\n\n"
            "Generate 10 realistic Q&A pairs about this topic. \n"
            "Requirements:\n"
            "1. Questions: Must sound like a real Indian SMB owner (Hindi, Hinglish, or English). Not formal.\n"
            "2. Answers: Accurate, cite the specific Act and Section number, under 120 words, and practical.\n"
            "3. Format: Output ONLY a JSON array of objects with fields: 'instruction', 'input', 'output'.\n"
            "4. Instruction field: Use the LexAI system prompt context.\n"
        )
        
        try:
            response = ollama.generate(
                model=self.model,
                system=self.system_prompt,
                prompt=prompt,
                format="json",
                options={"temperature": 0.8} # Higher temperature for variety in questions
            )
            
            data = json.loads(response['response'])
            return data
        except Exception as e:
            print(f"Generation error: {e}")
            return []

    def save_dataset(self, data, filename="db/synthetic_dataset.json"):
        """Saves or appends to a local training dataset."""
        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                pass
        
        # Append new pairs
        existing_data.extend(data)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
        
        return len(existing_data)

_generator_instance = None

def get_generator():
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SyntheticDataGenerator()
    return _generator_instance
