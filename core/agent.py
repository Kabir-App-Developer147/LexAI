from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from core.rag import load_rag_engine
from core.search import internet_search
from core.editor import get_editor
import os

from core.establishment import get_establishment

def create_lexai_agent(role="employee"):
    # 1. Load the local RAG engine and Establishment
    query_engine = load_rag_engine()
    est = get_establishment()
    profile_context = est.get_context_block()
    
    # 2. Define Tools
    
    # Tool for local documents
    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="local_knowledge",
            description="Use this tool to find information from the user's local documents."
        )
    )
    
    # Tool for internet search
    search_tool = FunctionTool.from_defaults(
        fn=internet_search,
        name="internet_search",
        description="Use this tool to search the internet for general knowledge, latest news, laws, or business trends not found in local documents."
    )
    
    # Tool for file editing (Safe Copy)
    editor = get_editor()
    
    def stage_file_change(filename: str, content: str):
        """Creates a modified copy of a file with new content."""
        if filename.endswith(".docx"):
            return editor.edit_docx_file(filename, content)
        else:
            return editor.edit_text_file(filename, content)

    edit_tool = FunctionTool.from_defaults(
        fn=stage_file_change,
        name="stage_file_change",
        description="Creates a modified COPY of a local file. Use this when the user asks to edit or update a document. It does NOT touch the original file yet."
    )

    def finalize_changes(filename: str):
        """Applies the modified copy to the main file."""
        return editor.apply_to_main(filename)

    finalize_tool = FunctionTool.from_defaults(
        fn=finalize_changes,
        name="finalize_changes",
        description="Overwrites the original file with the modified copy. ONLY use this when the user explicitly says to apply changes to the main/original file."
    )

    # 3. Initialize Agent with optimized LLM settings
    llm = Ollama(
        model="gemma3:4b", 
        request_timeout=120.0,
        additional_kwargs={
            "num_ctx": 4096,
            "temperature": 0.1, # Lower for agentic reliability
            "num_gpu": 1,
            "f16_kv": True
        }
    )
    
    agent = ReActAgent.from_tools(
        [rag_tool, search_tool, edit_tool, finalize_tool],
        llm=llm,
        verbose=True,
        max_iterations=5, # Prevent looping on medium-spec mobile
        context=(
            "# IDENTITY & SILENT MEMORY\n"
            "You are LexAI, a persistent, expert business assistant. Your processing is 100% private and offline, but you can use the internet for live data.\n"
            "Do NOT repeat the profile back—use it SILENTLY to give relevant answers.\n"
            "Primary Language: English.\n\n"
            f"# ACCESS CONTROL\n"
            f"Current User Role: {role}\n"
            "STRICT RULES:\n"
            "- if role is employee: Answer only HR and leave questions. NEVER reveal financial figures, contracts, or others' data.\n"
            "- if role is sales_staff: Discuss only customer documents, quotes, and pricing. NO internal financials or legal docs.\n"
            "- if role is accountant: Full access to financials and GST, but NO employee personal data or HR files.\n"
            "- if role is owner: No restrictions.\n"
            "IF OUTSIDE SCOPE: Respond ONLY with 'This information is not available for your access level' and do not hint at restricted data.\n\n"
            "# BEST EXPERIENCE: PROACTIVE SEARCH\n"
            "To provide the best experience, proactively use 'internet_search' to get latest laws, GST rules, news, and market trends.\n"
            "ALWAYS combine 'local_knowledge' (your private files) with 'internet_search' (live web) to give the most complete and up-to-date advice.\n\n"
            "# OPERATIONAL RULES\n"
            "1. DYNAMIC LEARNING: Note new user info for profile updates.\n"
            "2. READY-TO-USE DRAFTING: Produce complete drafts, not templates.\n"
            "3. LANGUAGE MIRRORING: Exactly mirror the user language (Hindi, Hinglish, English).\n\n"
            "# EXPERTISE & INTEGRITY\n"
            "1. EXPERT: Deep knowledge of GST, MSME, and Indian Business Acts (Section citations required).\n"
            "2. ZERO FABRICATION: Never generate fake figures.\n"
        )
    )

    return agent

if __name__ == "__main__":
    agent = create_lexai_agent()
    # Test integrated capability
    response = agent.chat("Check my documents for any travel plans and tell me the current weather at that destination from the internet.")
    print(response)
