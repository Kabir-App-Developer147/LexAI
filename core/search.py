from duckduckgo_search import DDGS
from llama_index.core import PromptTemplate
from core.llm import get_llm_manager

def internet_search(query: str, max_results: int = 5):
    """
    Performs a DuckDuckGo search and returns a summarized answer.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return "I tried searching the internet but couldn't find anything relevant."
        
        # Format the search results for the LLM
        context_str = ""
        for i, res in enumerate(results):
            context_str += f"[{i+1}] Source: {res.get('href')}\nContent: {res.get('body')}\n\n"
        
        llm = get_llm_manager()
        
        # Create a search-specific prompt
        search_prompt = (
            "You are LexAI, an expert business assistant for Indian business owners.\n"
            "Identity: Tum ek expert, vishwasaniya business sahayak ho. Primary Language: English.\n"
            "Tumne internet search kiya hai kyunki local data insufficient tha. Jawab niche diye gaye results ke aadhar par do.\n\n"
            "Search Results:\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "RULES:\n"
            "1. LANGUAGE MIRRORING: Hamesha usi bhasha (Hindi, Hinglish, English) mein jawab do jisme user ne pucha ho. Primary language remains English.\n"
            "2. EXPERTISE: Use your deep knowledge of Indian Laws (GST, Companies Act, etc.) to interpret these results.\n"
            "3. CITATIONS: Always cite sources [1] and mention specific SECTION NUMBERS if found.\n"
            "4. FALLBACK: Agar sahi jawab nahi mil raha online, toh saaf bol do.\n\n"
            f"Query: {query}\n"
            "Answer: "
        )
        
        # Using chat method with the prompt as the user message
        # We don't want to pass full history here to keep it focused on the search
        response, _ = llm.chat(search_prompt, [])
        return response
        
    except Exception as e:
        print(f"Search error: {e}")
        return f"Sorry, I ran into a technical glitch while searching the web: {e}"

if __name__ == "__main__":
    # Test search
    print(internet_search("Latest GST rules for small businesses in India 2025"))
