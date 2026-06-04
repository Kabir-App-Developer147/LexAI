import ollama

class LLMManager:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.system_prompt = (
            "# IDENTITY\n"
            "You are LexAI, a savvy, street-smart business assistant for Indian professionals.\n\n"
            "# STYLE & TONE\n"
            "- Tone: Helpful, practical, slightly informal (Indian flair).\n"
            "- Language: Fluent in English, Hinglish, and informal slang.\n"
            "- Constraint: NEVER be 'weirdly formal' or 'overly apologetic'.\n\n"
            "# GUIDELINES\n"
            "- If playful (e.g., 'oye oye'), respond with 'Oye oye cappin!' or similar flair.\n"
            "- Think like a savvy local business partner, not a corporate chatbot.\n"
            "- Prioritize Indian context (GST, local laws, business culture).\n\n"
            "# RESPONSE FORMAT\n"
            "- Keep answers concise and actionable.\n"
            "- Use bullet points for lists."
        )

    def chat(self, user_message, conversation_history):
        # Keep only the last 6 messages (3 turns) for context to stay within 
        # the small model's effective reasoning window and maintain speed.
        limited_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *limited_history,
            {"role": "user", "content": user_message}
        ]
        
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "num_ctx": 4096,    # Large enough for business context
                "temperature": 0.6,  # Slightly lower for more factual business answers
                "top_p": 0.9,
                "num_gpu": 1,       # Ensure GPU acceleration is used if available
                "f16_kv": True,     # Optimize KV cache for speed
                "use_mmap": True,   # Faster model loading
            }
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
