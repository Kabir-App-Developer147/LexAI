from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from core.rag import load_rag_engine
from core.search import internet_search
from core.editor import get_editor
import os

def create_lexai_agent():
    # 1. Load the local RAG engine
    query_engine = load_rag_engine()
    
    # 2. Define Tools
    
    # Tool for local documents
    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="local_knowledge",
            description="Use this tool to find information from the user's local documents (PDFs, DOCX, TXT). This contains sensitive business data."
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

    # 3. Initialize Agent
    llm = Ollama(model="gemma3:4b", request_timeout=120.0)
    
    agent = ReActAgent.from_tools(
        [rag_tool, search_tool, edit_tool, finalize_tool],
        llm=llm,
        verbose=True,
        context=(
            "You are LexAI, the street-smart Indian business assistant. "
            "You have access to local documents, the internet, and a safe file editor. "
            "When a user asks something, you can use local_knowledge and internet_search TOGETHER to provide a comprehensive answer. "
            "If a user asks to edit a file, use stage_file_change first. NEVER overwrite a main file unless specifically told to 'apply changes' or 'update the original'."
            "Maintain your friendly, practical, and slightly informal Indian flair."
        )
    )
    
    return agent

if __name__ == "__main__":
    agent = create_lexai_agent()
    # Test integrated capability
    response = agent.chat("Check my documents for any travel plans and tell me the current weather at that destination from the internet.")
    print(response)
