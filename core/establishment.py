import os
import json

# Comprehensive Localized Data for Indian Business Environment
INDIAN_MARKET_DATA = {
    "business_entities": [
        "Sole Proprietorship", 
        "Partnership Firm", 
        "Private Limited Company (Pvt Ltd)", 
        "Limited Liability Partnership (LLP)", 
        "One Person Company (OPC)"
    ],
    "tax_milestones": {
        "GST": "GSTR-1 (Monthly/Quarterly), GSTR-3B (Monthly)",
        "Income Tax": "Advance Tax installments (June, Sept, Dec, March)",
        "Audit": "Tax Audit deadline (Sept/Oct depending on year)"
    },
    "common_vocabulary": {
        "Bahi Khata": "Traditional ledger/accounting book",
        "Udhaar": "Credit or Debt records",
        "Challan": "Official payment receipt or invoice",
        "Galla": "Daily cash box",
        "Pukka Bill": "GST-compliant tax invoice",
        "Kachha Bill": "Estimates or non-tax memos"
    },
    "local_market_dynamics": {
        "Maharashtra": "Strong emphasis on Gudi Padwa sales and Diwali bonuses.",
        "Gujarat": "High focus on business networking and 'Vyapar' culture.",
        "Tamil Nadu": "Heavy compliance focus around Pongal and New Year.",
        "Delhi/North": "High seasonal trade fluctuations around winter."
    }
}

class Establishment:
    """
    Handles the specific environment and identity of the user's business.
    """
    def __init__(self, profile_path="db/establishment_profile.json"):
        self.profile_path = profile_path
        self._ensure_db_folder()
        self.profile = self._load_profile()

    def _ensure_db_folder(self):
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)

    def _load_profile(self):
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # Default Savvy Profile with Memory Fields
        default_profile = {
            "name": "Suresh & Sons Enterprises",
            "owner": "Suresh Kumar",
            "location": {"city": "Mumbai", "state": "Maharashtra", "pincode": "400001"},
            "type": "General Trade & Distribution",
            "gst_status": "Regular",
            "memory": {
                "summary": "First time user setup.",
                "corrections": [],
                "learned_facts": []
            },
            "environment": {
                "language_preference": "Hinglish",
                "currency": "INR"
            }
        }
        self.save_profile(default_profile)
        return default_profile

    def get_silent_memory_block(self):
        """Returns the context for the Persistent Memory Assistant layer."""
        p = self.profile
        mem = p.get('memory', {})
        
        return (
            f"\n# PERSISTENT BUSINESS MEMORY (SILENT)\n"
            f"- User Profile: {json.dumps(p, indent=2)}\n"
            f"- Memory Summary: {mem.get('summary', 'No previous history')}\n"
            f"- Past Corrections: {', '.join(mem.get('corrections', []))}\n"
            f"- Learned Facts: {', '.join(mem.get('learned_facts', []))}\n"
            "INSTRUCTION: Use this memory SILENTLY. Do not repeat the profile. "
            "If the user gives new business info, note it for profile updates."
        )

    def update_memory_from_chat(self, user_msg, assistant_reply):
        """
        ACTUAL CODE: Programmatically distills new facts from chat and updates JSON.
        This ensures memory is persistent and not just a prompt instruction.
        """
        import ollama
        distill_prompt = (
            "Given the user message and assistant reply, identify any NEW facts about the user's business "
            "or any corrections they made to existing info. \n\n"
            f"User: {user_msg}\nAssistant: {assistant_reply}\n\n"
            "Output ONLY a JSON object with: { 'new_facts': [], 'corrections': [] }. No explanation."
        )
        
        try:
            res = ollama.generate(model="gemma3:4b", prompt=distill_prompt, format="json")
            updates = json.loads(res['response'])
            
            modified = False
            if updates.get('new_facts'):
                self.profile['memory']['learned_facts'].extend(updates['new_facts'])
                # Keep it clean: only unique facts
                self.profile['memory']['learned_facts'] = list(set(self.profile['memory']['learned_facts']))
                modified = True
            
            if updates.get('corrections'):
                self.profile['memory']['corrections'].extend(updates['corrections'])
                modified = True
            
            if modified:
                print(f"DEBUG: Persistent Memory Updated with {len(updates.get('new_facts', []))} facts.")
                self.save_profile(self.profile)
                
        except Exception as e:
            print(f"Memory update error: {e}")

    def save_profile(self, data):
        with open(self.profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def get_context_block(self):
        """Returns a string describing the environment for LLM grounding."""
        p = self.profile
        loc = p['location']
        env = p['environment']
        state_note = INDIAN_MARKET_DATA['local_market_dynamics'].get(loc['state'], "")
        
        return (
            f"\n# ESTABLISHMENT PROFILE\n"
            f"- Business Name: {p['name']}\n"
            f"- Owner/User: {p['owner']}\n"
            f"- Location: {loc['city']}, {loc['state']}\n"
            f"- Industry: {p['type']}\n"
            f"- GST Regime: {p['gst_status']}\n"
            f"- Market Note: {state_note}\n"
            f"- Local Terms: {', '.join(INDIAN_MARKET_DATA['common_vocabulary'].keys())}\n"
        )

    def generate_welcome_greeting(self):
        """Generates a localized welcome message for the UI."""
        p = self.profile
        return (
            f"Namaste, {p['owner']} ji! Welcome to {p['name']}.\n"
            f"Everything in {p['location']['city']} looks ready. Your {p['type']} files are synced.\n"
            "Aaj kya help karoon? (How can I help you today?)"
        )

# Global Accessor for Lazy Init
_establishment_instance = None

def get_establishment():
    global _establishment_instance
    if _establishment_instance is None:
        _establishment_instance = Establishment()
    return _establishment_instance
