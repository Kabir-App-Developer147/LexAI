import ollama

class LLMManager:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.system_prompt = (
            "You are LexAI, a helpful assistant for small businesses and professionals in India. "
            "Be concise and practical. Always consider Indian laws, GST, and local context where relevant."
        )

    def chat(self, user_message, conversation_history):
        messages = [
            {"role": "system", "content": self.system_prompt},
            *conversation_history,
            {"role": "user", "content": user_message}
        ]
        
        response = ollama.chat(
            model=self.model,
            messages=messages
        )
        
        reply = response['message']['content']
        
        # We don't update history here to keep the manager stateless/flexible, 
        # but for compatibility with main.py's current structure, we return it.
        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply}
        ]
        
        return reply, updated_history

llm = LLMManager()
