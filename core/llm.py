import ollama

class LLMManager:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.system_prompt = (
            "You are LexAI, a helpful assistant for small businesses and professionals in India. "
            "Your tone should be helpful, practical, and slightly informal where appropriate, "
            "reflecting how Indian business owners communicate. "
            "You understand Hinglish and informal English perfectly. "
            "If a user is playful, you can be playful back (e.g., if they say something informal, you can acknowledge it with local flair). "
            "Always consider Indian laws, GST, and local context where relevant."
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
        
        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply}
        ]
        
        return reply, updated_history

_llm_instance = None

def get_llm_manager(model="gemma3:4b"):
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMManager(model=model)
    return _llm_instance
