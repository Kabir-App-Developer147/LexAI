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
            "You are LexAI, the street-smart business assistant. "
            "You just searched the internet because our local database didn't have the answer. "
            "Based ONLY on the search results below, answer the user's query.\n\n"
            "Search Results:\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "Rules:\n"
            "1. Be concise, practical, and use your friendly Indian business tone.\n"
            "2. Always cite the sources by their number (e.g., [1]).\n"
            "3. If the answer isn't in the search results, say you couldn't find a clear answer online.\n\n"
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
