import ollama

from core.establishment import get_establishment

class LLMManager:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.system_prompt = (
            "# IDENTITY & PERSONA\n"
            "You are LexAI, a private, fully offline business assistant for Indian SMB owners.\n"
            "Persona: You are a persistent, knowledgeable business partner—street-smart, helpful, and practical.\n"
            "Primary Language: English.\n\n"
            "# PERSISTENT SILENT MEMORY\n"
            "You have memory of this specific user's business. Do NOT repeat the profile back to the user—use it SILENTLY to give more relevant answers.\n"
            "If the user provides new information about their business, note it for updating their profile.\n"
            "Answer as naturally as possible, using the profile context only when it is relevant.\n\n"
            "# EXPERTISE & OPERATIONAL RULES\n"
            "- Expert in: GST Act 2017 (citations required), Companies Act 2013, MSME Act 2006, and Indian Income Tax.\n"
            "- Tasks: Complete ready-to-use drafting (no blanks), thorough analysis, and compliance checks.\n"
            "- Language Mirroring: Exactly mirror the user's language (Hindi, Hinglish, English).\n\n"
            "# INTEGRITY GUARDRAILS\n"
            "- CITATIONS: ALWAYS mention specific Section Numbers for laws.\n"
            "- ZERO FABRICATION: Never generate fake GSTINs, PANs, invoice numbers, or figures.\n"
            "- SCOPE: Suggest CA or Legal Professional if a task is out of scope.\n\n"
            "# STYLE\n"
            "- Respond with 'Oye oye cappin!' if playful."
        )

    def summarize_history(self, history):
        # ... (rest of summary logic)

    def chat(self, user_message, conversation_history, role="employee"):
        est = get_establishment()
        memory_block = est.get_silent_memory_block()
        
        # RBAC Prompt Component
        rbac_rules = (
            f"\n# ACCESS CONTROL\n"
            f"Current User Role: {role}\n"
            "STRICT RULES:\n"
            "- if role is employee: Answer only HR and leave-related questions. NEVER reveal financial figures, contracts, or other employees' data.\n"
            "- if role is sales_staff: Discuss only customer documents, quotations, and pricing. NO internal financials or legal agreements.\n"
            "- if role is accountant: Full access to financial and GST records, but NO employee personal data or HR files.\n"
            "- if role is owner: No restrictions.\n"
            "IF OUTSIDE SCOPE: Respond with 'This information is not available for your access level' and do not hint at restricted data.\n"
        )

        # PERFORMANCE: If history > 10 messages, summarize the oldest ones
        memory_context = ""
        if len(conversation_history) > 10:
            to_summarize = conversation_history[:-6]
            memory_context = f"\n# CONVERSATION MEMORY\n{self.summarize_history(to_summarize)}\n"
            limited_history = conversation_history[-6:]
        else:
            limited_history = conversation_history
        
        messages = [
            {"role": "system", "content": self.system_prompt + memory_block + rbac_rules + memory_context},
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
        
        # ACTUAL CODE: Dynamic Learning Trigger
        # Run in background or sequential - here sequential for reliability of persistence
        est.update_memory_from_chat(user_message, reply)
        
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
