from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os
import shutil
from typing import List
from contextlib import asynccontextmanager
import anyio

# Ensure core modules are importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import load_rag_engine, ask
from core.llm import get_llm_manager

# Global instances
engine = None
llm_manager = None
DATA_DIR = "data"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global engine, llm_manager
    print("Loading AI Engines...")
    # Load engines in threads to keep startup responsive
    engine = await anyio.to_thread.run_sync(load_rag_engine)
    llm_manager = await anyio.to_thread.run_sync(get_llm_manager)
    print("Ready.")
    yield
    # Shutdown logic (if any)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class Question(BaseModel):
    text: str
    history: List[dict] = []

@app.post("/ask")
async def ask_question(q: Question):
    if not engine or not llm_manager:
        raise HTTPException(status_code=503, detail="AI engines not initialized")
    
    # 1. Try RAG first
    rag_response = await anyio.to_thread.run_sync(ask, engine, q.text)
    
    # If RAG is unsure (LlamaIndex usually says "Empty Response" or similar if set up that way,
    # or we can check for the "don't know" phrase from our prompt)
    unsure_phrases = ["don't know", "not mentioned in the context", "no context information"]
    is_unsure = any(phrase in rag_response.lower() for phrase in unsure_phrases)
    
    if is_unsure:
        # 2. Fallback to LLM for general chat/help
        print(f"RAG unsure, falling back to LLM for: {q.text}")
        llm_response, updated_history = await anyio.to_thread.run_sync(
            llm_manager.chat, q.text, q.history
        )
        return {
            "answer": llm_response,
            "source": "LLM",
            "history": updated_history
        }
    
    # 3. Return RAG response
    # For now, let's keep the history update simple for RAG too
    updated_history = q.history + [
        {"role": "user", "content": q.text},
        {"role": "assistant", "content": rag_response}
    ]
    
    return {
        "answer": rag_response,
        "source": "RAG",
        "history": updated_history
    }

@app.get("/files")
async def list_files():
    """List all files in the data directory."""
    if not os.path.exists(DATA_DIR):
        return []
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    return files

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "status": "uploaded"}

@app.post("/sync")
async def sync_index():
    """Manually trigger a sync of the RAG index."""
    global engine
    print("Syncing RAG engine...")
    engine = await anyio.to_thread.run_sync(load_rag_engine)
    return {"status": "synced"}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LexAI - Business Intelligence</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            body { font-family: 'Inter', sans-serif; }
            .chat-container::-webkit-scrollbar { width: 6px; }
            .chat-container::-webkit-scrollbar-thumb { background-color: #475569; border-radius: 10px; }
            .typing-dot { animation: typing 1.4s infinite; }
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-4px); } }
        </style>
    </head>
    <body class="bg-slate-900 text-slate-100 h-screen flex overflow-hidden">

        <!-- Sidebar -->
        <aside class="w-72 bg-slate-950 border-r border-slate-800 flex flex-col hidden md:flex">
            <div class="p-6">
                <div class="flex items-center gap-3 mb-8">
                    <div class="bg-indigo-600 p-2 rounded-lg">
                        <i data-lucide="scale" class="w-6 h-6 text-white"></i>
                    </div>
                    <h1 class="text-xl font-bold tracking-tight">LexAI</h1>
                </div>
                
                <button onclick="newChat()" class="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 py-2.5 px-4 rounded-xl flex items-center justify-center gap-2 transition-all border border-slate-700 mb-6">
                    <i data-lucide="plus" class="w-4 h-4"></i>
                    <span class="font-medium text-sm">New Chat</span>
                </button>

                <div class="mb-4">
                    <h2 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-2">Knowledge Base</h2>
                    <div id="file-list" class="space-y-1 overflow-y-auto max-h-[40vh]">
                        <!-- Files populated here -->
                    </div>
                </div>
            </div>

            <div class="mt-auto p-6 border-t border-slate-900">
                <div class="relative group">
                    <input type="file" id="file-upload" class="hidden" onchange="handleUpload(this)">
                    <label for="file-upload" class="flex items-center gap-3 text-slate-400 hover:text-white cursor-pointer transition-colors p-2 rounded-lg hover:bg-slate-900">
                        <i data-lucide="upload-cloud" class="w-5 h-5"></i>
                        <span class="text-sm font-medium">Upload Document</span>
                    </label>
                </div>
                <button onclick="syncIndex()" class="flex items-center gap-3 text-slate-400 hover:text-white w-full text-left p-2 rounded-lg hover:bg-slate-900 mt-2 transition-colors">
                    <i data-lucide="refresh-cw" class="w-5 h-5" id="sync-icon"></i>
                    <span class="text-sm font-medium">Sync Index</span>
                </button>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="flex-1 flex flex-col relative bg-slate-900">
            <!-- Header -->
            <header class="h-16 border-b border-slate-800 flex items-center px-8 justify-between bg-slate-900/50 backdrop-blur-md sticky top-0 z-10">
                <div class="flex items-center gap-4">
                    <button class="md:hidden text-slate-400"><i data-lucide="menu"></i></button>
                    <h2 class="font-semibold text-slate-200">Local Business Assistant</h2>
                </div>
                <div class="flex items-center gap-2 text-xs text-slate-500 bg-slate-800/50 px-3 py-1.5 rounded-full border border-slate-700">
                    <div class="w-2 h-2 rounded-full bg-green-500"></div>
                    <span>gemma3:4b</span>
                </div>
            </header>

            <!-- Chat Area -->
            <div id="chat-messages" class="flex-1 overflow-y-auto p-8 space-y-8 chat-container">
                <!-- Welcome Message -->
                <div class="max-w-3xl mx-auto text-center py-12 space-y-4">
                    <div class="bg-indigo-600/10 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6">
                        <i data-lucide="bot" class="w-8 h-8 text-indigo-500"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-white">Namaste! I'm LexAI</h3>
                    <p class="text-slate-400 max-w-md mx-auto">Your local business expert. Ask me anything about your documents, GST, or Indian business laws.</p>
                </div>
            </div>

            <!-- Input Area -->
            <div class="p-6 bg-gradient-to-t from-slate-900 via-slate-900 to-transparent">
                <div class="max-w-3xl mx-auto relative group">
                    <textarea 
                        id="user-input"
                        rows="1"
                        placeholder="Type your message..."
                        class="w-full bg-slate-800 border border-slate-700 rounded-2xl py-4 pl-6 pr-14 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all resize-none shadow-2xl"
                        onkeydown="handleKey(event)"
                    ></textarea>
                    <button 
                        onclick="sendMessage()"
                        class="absolute right-3 bottom-3 p-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                        id="send-btn"
                    >
                        <i data-lucide="send" class="w-5 h-5"></i>
                    </button>
                </div>
                <p class="text-center text-[10px] text-slate-600 mt-4 uppercase tracking-widest font-medium">Powering Small Businesses in India</p>
            </div>
        </main>

        <script>
            // Initialize Lucide Icons
            lucide.createIcons();

            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const fileList = document.getElementById('file-list');
            let conversationHistory = [];

            async function fetchFiles() {
                const res = await fetch('/files');
                const files = await res.json();
                fileList.innerHTML = files.map(f => `
                    <div class="flex items-center gap-3 p-2 rounded-lg text-slate-400 hover:bg-slate-900 hover:text-slate-200 transition-colors group">
                        <i data-lucide="file-text" class="w-4 h-4 text-slate-500"></i>
                        <span class="text-sm truncate font-medium">${f}</span>
                    </div>
                `).join('');
                lucide.createIcons();
            }

            async function handleUpload(input) {
                if (!input.files[0]) return;
                const formData = new FormData();
                formData.append('file', input.files[0]);
                
                try {
                    await fetch('/upload', { method: 'POST', body: formData });
                    await fetchFiles();
                    await syncIndex();
                } catch (e) { console.error("Upload failed", e); }
            }

            async function syncIndex() {
                const icon = document.getElementById('sync-icon');
                icon.classList.add('animate-spin');
                try {
                    await fetch('/sync', { method: 'POST' });
                    setTimeout(() => icon.classList.remove('animate-spin'), 1000);
                } catch (e) { icon.classList.remove('animate-spin'); }
            }

            function addMessage(text, isAi = false, source = null) {
                const welcomeMsg = document.querySelector('.max-w-3xl.mx-auto.text-center');
                if (welcomeMsg) welcomeMsg.remove();

                const div = document.createElement('div');
                div.className = `flex ${isAi ? 'justify-start' : 'justify-end'} animate-in fade-in slide-in-from-bottom-2 duration-300`;
                
                const sourceTag = source ? `<span class="text-[10px] opacity-50 block mt-1 uppercase tracking-tighter">${source}</span>` : '';
                
                const content = `
                    <div class="flex gap-4 max-w-[85%] ${isAi ? '' : 'flex-row-reverse'}">
                        <div class="w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center ${isAi ? 'bg-indigo-600/20 text-indigo-500' : 'bg-slate-700 text-slate-300'}">
                            <i data-lucide="${isAi ? 'bot' : 'user'}" class="w-5 h-5"></i>
                        </div>
                        <div class="${isAi ? 'bg-slate-800 text-slate-200' : 'bg-indigo-600 text-white'} p-4 rounded-2xl shadow-md text-sm leading-relaxed">
                            ${text.replace(/\\n/g, '<br>')}
                            ${sourceTag}
                        </div>
                    </div>
                `;
                div.innerHTML = content;
                chatMessages.appendChild(div);
                lucide.createIcons();
                chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
            }

            function showTyping() {
                const div = document.createElement('div');
                div.id = 'typing-indicator';
                div.className = 'flex justify-start';
                div.innerHTML = `
                    <div class="flex gap-4 max-w-[85%]">
                        <div class="w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center bg-indigo-600/20 text-indigo-500">
                            <i data-lucide="bot" class="w-5 h-5"></i>
                        </div>
                        <div class="bg-slate-800 p-4 rounded-2xl flex gap-1 items-center">
                            <span class="w-1.5 h-1.5 bg-slate-500 rounded-full typing-dot"></span>
                            <span class="w-1.5 h-1.5 bg-slate-500 rounded-full typing-dot"></span>
                            <span class="w-1.5 h-1.5 bg-slate-500 rounded-full typing-dot"></span>
                        </div>
                    </div>
                `;
                chatMessages.appendChild(div);
                lucide.createIcons();
                chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
            }

            async function sendMessage() {
                const text = userInput.value.trim();
                if (!text) return;

                userInput.value = '';
                userInput.style.height = 'auto';
                addMessage(text, false);
                
                showTyping();
                
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            text: text,
                            history: conversationHistory
                        })
                    });
                    const data = await res.json();
                    document.getElementById('typing-indicator').remove();
                    addMessage(data.answer, true, data.source);
                    conversationHistory = data.history;
                } catch (e) {
                    document.getElementById('typing-indicator').remove();
                    addMessage("Sorry, something went wrong. Please check your connection.", true);
                }
            }

            function handleKey(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
                // Auto-resize textarea
                e.target.style.height = 'auto';
                e.target.style.height = (e.target.scrollHeight) + 'px';
            }

            function newChat() {
                conversationHistory = [];
                chatMessages.innerHTML = `
                    <div class="max-w-3xl mx-auto text-center py-12 space-y-4">
                        <div class="bg-indigo-600/10 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6">
                            <i data-lucide="bot" class="w-8 h-8 text-indigo-500"></i>
                        </div>
                        <h3 class="text-2xl font-bold text-white">New Conversation Started</h3>
                        <p class="text-slate-400 max-w-md mx-auto">How can I help you today?</p>
                    </div>
                `;
                lucide.createIcons();
            }

            // Initial load
            fetchFiles();
        </script>
    </body>
    </html>
    """
