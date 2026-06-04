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

from core.agent import create_lexai_agent
from core.establishment import get_establishment
from core.auth import get_auth_manager
from core.synthetic import get_generator

# Global instances
agent = None
DATA_DIR = "data"

# ... (FastAPI setup)

class TopicRequest(BaseModel):
    topic: str

@app.post("/generate-data")
async def generate_training_data(req: TopicRequest):
    """Generates synthetic Q&A pairs for fine-tuning."""
    gen = get_generator()
    pairs = await anyio.to_thread.run_sync(gen.generate_pairs, req.topic)
    if not pairs:
        raise HTTPException(status_code=500, detail="Failed to generate data")
    
    count = gen.save_dataset(pairs)
    return {"message": f"Generated 10 pairs. Dataset now has {count} items.", "pairs": pairs}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global agent
    print("Loading LexAI Agent...")
    # Initialize the integrated agent
    agent = await anyio.to_thread.run_sync(create_lexai_agent)
    print("Ready.")
    yield
    # Shutdown logic
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class UserAuth(BaseModel):
    username: str
    password: str
    role: str = "employee"

class Question(BaseModel):
    text: str
    history: List[dict] = []
    token: str = None # Pass token for role identification

@app.post("/signup")
async def signup(user: UserAuth):
    auth = get_auth_manager()
    success, msg = auth.signup(user.username, user.password, user.role)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.post("/login")
async def login(user: UserAuth):
    auth = get_auth_manager()
    token, msg = auth.login(user.username, user.password)
    if not token:
        raise HTTPException(status_code=401, detail=msg)
    return {"token": token, "message": msg}

# ... (greeting endpoint)

@app.post("/ask")
async def ask_question(q: Question):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # 1. Identify Role from Token
    role = "employee"
    if q.token:
        try:
            from core.auth import SECRET_KEY, ALGORITHM
            from jose import jwt
            payload = jwt.decode(q.token, SECRET_KEY, algorithms=[ALGORITHM])
            role = payload.get("role", "employee")
        except:
            pass

    # 2. Re-initialize agent if role changed (or pass role to chat if supported)
    # For ReActAgent, we need to pass role during creation or manage multiple agents.
    # To keep it simple and efficient, we'll create a role-specific agent call.
    role_agent = await anyio.to_thread.run_sync(create_lexai_agent, role)
    
    # Use the Agent to decide which tools to use
    response = await anyio.to_thread.run_sync(role_agent.chat, q.text)
    answer = str(response)
    
    # Source detection
    source = "Agent"
    if "Source: [" in answer: source = "Internet"
    elif "As per [" in answer or "Checking [" in answer: source = "RAG"
    
    return {
        "answer": answer,
        "source": source,
        "history": q.history + [{"role": "user", "content": q.text}, {"role": "assistant", "content": answer}]
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
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <title>LexAI Mobile</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
            body { 
                font-family: 'Plus Jakarta Sans', sans-serif; 
                -webkit-tap-highlight-color: transparent;
                overscroll-behavior-y: contain;
            }
            .chat-container::-webkit-scrollbar { display: none; }
            .chat-container { -ms-overflow-style: none; scrollbar-width: none; }
            
            .safe-bottom { padding-bottom: env(safe-area-inset-bottom); }
            .safe-top { padding-top: env(safe-area-inset-top); }
            
            .tab-active { color: #6366f1; }
            .tab-inactive { color: #94a3b8; }
            
            @keyframes slideUp {
                from { transform: translateY(20px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .msg-animate { animation: slideUp 0.3s ease-out forwards; }
            
            .typing-dot { animation: typing 1.4s infinite; }
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-4px); } }
        </style>
    </head>
    <body class="bg-slate-50 text-slate-900 h-screen flex flex-col overflow-hidden">

        <!-- Auth Screen (Overlay) -->
        <div id="auth-screen" class="fixed inset-0 z-[100] bg-slate-50 flex flex-col items-center justify-center p-8 safe-top safe-bottom">
            <div class="w-full max-w-md space-y-8 animate-in fade-in zoom-in duration-500">
                <div class="text-center space-y-4">
                    <div class="bg-indigo-600 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto shadow-2xl shadow-indigo-200">
                        <i data-lucide="lock" class="w-10 h-10 text-white"></i>
                    </div>
                    <h2 class="text-3xl font-extrabold text-slate-900">Secure LexAI</h2>
                    <p class="text-slate-500 font-medium">Your business data, locally protected.</p>
                </div>

                <div class="bg-white p-8 rounded-3xl shadow-xl shadow-slate-200 border border-slate-100 space-y-6">
                    <div class="space-y-4">
                        <div class="space-y-2">
                            <label class="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Username</label>
                            <input type="text" id="auth-username" class="w-full bg-slate-50 border border-slate-200 rounded-2xl py-4 px-5 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all text-base" placeholder="Enter username">
                        </div>
                        <div class="space-y-2">
                            <label class="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Password</label>
                            <input type="password" id="auth-password" class="w-full bg-slate-50 border border-slate-200 rounded-2xl py-4 px-5 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all text-base" placeholder="••••••••">
                        </div>
                        <div class="space-y-2">
                            <label class="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Role (for Signup)</label>
                            <select id="auth-role" class="w-full bg-slate-50 border border-slate-200 rounded-2xl py-4 px-5 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all text-base">
                                <option value="employee">Employee</option>
                                <option value="sales_staff">Sales Staff</option>
                                <option value="accountant">Accountant</option>
                                <option value="owner">Owner</option>
                            </select>
                        </div>
                    </div>

                    <div class="flex flex-col gap-3">
                        <button onclick="handleAuth('login')" id="login-btn" class="w-full bg-indigo-600 text-white py-4 rounded-2xl font-bold shadow-lg shadow-indigo-100 active:scale-95 transition-all">
                            Login to Establishment
                        </button>
                        <button onclick="handleAuth('signup')" id="signup-btn" class="w-full bg-slate-100 text-slate-600 py-4 rounded-2xl font-bold active:scale-95 transition-all">
                            Create Local Account
                        </button>
                    </div>
                    
                    <p id="auth-error" class="text-center text-sm font-semibold text-rose-500 hidden animate-pulse"></p>
                </div>
                
                <p class="text-center text-[10px] text-slate-400 uppercase tracking-[0.2em] font-bold">100% Local Authentication</p>
            </div>
        </div>

        <!-- Mobile Header -->
        <header class="bg-white/80 backdrop-blur-md border-b border-slate-100 px-6 py-4 flex items-center justify-between sticky top-0 z-30 safe-top">
            <div class="flex items-center gap-3">
                <div class="bg-indigo-600 p-2 rounded-xl shadow-lg shadow-indigo-200">
                    <i data-lucide="scale" class="w-5 h-5 text-white"></i>
                </div>
                <h1 class="text-xl font-bold tracking-tight text-slate-800">LexAI</h1>
            </div>
            <button onclick="newChat()" class="p-2 text-slate-400 hover:text-indigo-600 transition-colors">
                <i data-lucide="rotate-ccw" class="w-5 h-5"></i>
            </button>
        </header>

        <!-- Main Content Area -->
        <main class="flex-1 relative overflow-hidden flex flex-col">
            
            <!-- Chat View -->
            <div id="chat-view" class="flex-1 overflow-y-auto p-4 space-y-4 chat-container pb-32">
                <div class="max-w-md mx-auto text-center py-10 space-y-4" id="welcome-screen">
                    <div class="bg-indigo-50 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-4 border border-indigo-100">
                        <i data-lucide="sparkles" class="w-10 h-10 text-indigo-500"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-slate-800">Namaste!</h3>
                    <p class="text-slate-500 px-6 leading-relaxed">I'm your savvy business assistant. Ask me about your documents or search the web.</p>
                </div>
                <div id="chat-messages" class="space-y-4"></div>
            </div>

            <!-- Knowledge Base View (Hidden by default) -->
            <div id="kb-view" class="hidden flex-1 overflow-y-auto p-6 space-y-6 bg-white">
                <div class="flex items-center justify-between">
                    <h2 class="text-xl font-bold text-slate-800">Your Files</h2>
                    <label for="file-upload" class="bg-indigo-600 text-white px-4 py-2 rounded-xl text-sm font-semibold shadow-md shadow-indigo-200 cursor-pointer">
                        Add New
                    </label>
                    <input type="file" id="file-upload" class="hidden" onchange="handleUpload(this)">
                </div>
                <div id="file-list" class="grid grid-cols-1 gap-3">
                    <!-- Files populated here -->
                </div>
                <button onclick="syncIndex()" class="w-full border border-slate-200 text-slate-600 py-3 rounded-xl flex items-center justify-center gap-2 font-medium">
                    <i data-lucide="refresh-cw" class="w-4 h-4" id="sync-icon"></i>
                    Sync Knowledge Base
                </button>
            </div>

            <!-- Dynamic Input Area (Only for Chat View) -->
            <div id="input-container" class="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-slate-50 via-slate-50/95 to-transparent z-20">
                <div class="max-w-2xl mx-auto flex items-end gap-2 bg-white p-2 rounded-2xl shadow-xl shadow-slate-200 border border-slate-100">
                    <textarea 
                        id="user-input"
                        rows="1"
                        placeholder="Type a message..."
                        class="flex-1 bg-transparent border-none py-3 px-3 text-slate-800 placeholder-slate-400 focus:outline-none resize-none text-base"
                        onkeydown="handleKey(event)"
                        oninput="this.style.height='auto';this.style.height=(this.scrollHeight)+'px'"
                    ></textarea>
                    <button 
                        onclick="sendMessage()"
                        id="send-btn"
                        class="bg-indigo-600 text-white p-3 rounded-xl shadow-lg shadow-indigo-100 disabled:opacity-50 transition-all active:scale-95"
                    >
                        <i data-lucide="arrow-up" class="w-5 h-5"></i>
                    </button>
                </div>
            </div>
        </main>

        <!-- Bottom Navigation -->
        <nav class="bg-white border-t border-slate-100 flex justify-around items-center py-3 safe-bottom z-30 shadow-[0_-4px_12px_rgba(0,0,0,0.02)]">
            <button onclick="switchTab('chat')" id="tab-chat" class="flex flex-col items-center gap-1 tab-active px-6 py-1">
                <i data-lucide="message-square" class="w-6 h-6"></i>
                <span class="text-[10px] font-bold uppercase tracking-widest">Chat</span>
            </button>
            <button onclick="switchTab('kb')" id="tab-kb" class="flex flex-col items-center gap-1 tab-inactive px-6 py-1">
                <i data-lucide="folder" class="w-6 h-6"></i>
                <span class="text-[10px] font-bold uppercase tracking-widest">Knowledge</span>
            </button>
        </nav>

        <script>
            lucide.createIcons();

            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const welcomeScreen = document.getElementById('welcome-screen');
            const fileList = document.getElementById('file-list');
            
            // AUTH & PERFORMANCE
            let conversationHistory = JSON.parse(localStorage.getItem('lexai_history') || '[]');
            let authToken = localStorage.getItem('lexai_token');

            // Initialize App
            if (authToken) {
                document.getElementById('auth-screen').classList.add('hidden');
                initApp();
            }

            async function handleAuth(type) {
                const u = document.getElementById('auth-username').value;
                const p = document.getElementById('auth-password').value;
                const err = document.getElementById('auth-error');
                
                if (!u || !p) {
                    err.innerText = "Fill all fields, boss!";
                    err.classList.remove('hidden');
                    return;
                }

                try {
                    const res = await fetch(`/${type}`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({username: u, password: p})
                    });
                    const data = await res.json();
                    
                    if (res.ok) {
                        if (type === 'login') {
                            localStorage.setItem('lexai_token', data.token);
                            document.getElementById('auth-screen').classList.add('hidden');
                            initApp();
                        } else {
                            err.innerText = "Account created! Now login.";
                            err.classList.remove('hidden');
                            err.className = "text-center text-sm font-semibold text-emerald-500";
                        }
                    } else {
                        err.innerText = data.detail || "Auth failed";
                        err.classList.remove('hidden');
                    }
                } catch (e) {
                    err.innerText = "Connection error";
                    err.classList.remove('hidden');
                }
            }

            function initApp() {
                if (conversationHistory.length > 0) {
                    conversationHistory.forEach(msg => addMessage(msg.content, msg.role === 'assistant'));
                } else {
                    fetch('/greeting')
                        .then(res => res.json())
                        .then(data => {
                            if (data.greeting) addMessage(data.greeting, true);
                        });
                }
                fetchFiles();
            }

            function switchTab(tab) {
                const isChat = tab === 'chat';
                document.getElementById('chat-view').classList.toggle('hidden', !isChat);
                document.getElementById('kb-view').classList.toggle('hidden', isChat);
                document.getElementById('input-container').classList.toggle('hidden', !isChat);
                
                document.getElementById('tab-chat').className = `flex flex-col items-center gap-1 ${isChat ? 'tab-active' : 'tab-inactive'} px-6 py-1`;
                document.getElementById('tab-kb').className = `flex flex-col items-center gap-1 ${!isChat ? 'tab-active' : 'tab-inactive'} px-6 py-1`;
                
                if (!isChat) fetchFiles();
                lucide.createIcons();
            }

            async function fetchFiles() {
                // INSTANT UI: Show cached files first, then update
                const cached = localStorage.getItem('lexai_files');
                if (cached) renderFiles(JSON.parse(cached));

                try {
                    const res = await fetch('/files');
                    const files = await res.json();
                    localStorage.setItem('lexai_files', JSON.stringify(files));
                    renderFiles(files);
                } catch (e) {}
            }

            function renderFiles(files) {
                fileList.innerHTML = files.map(f => `
                    <div class="flex items-center justify-between p-4 bg-slate-50 rounded-2xl border border-slate-100 group">
                        <div class="flex items-center gap-3">
                            <div class="bg-white p-2 rounded-lg border border-slate-200">
                                <i data-lucide="file-text" class="w-5 h-5 text-indigo-500"></i>
                            </div>
                            <span class="text-sm font-semibold text-slate-700 truncate max-w-[150px]">${f}</span>
                        </div>
                        <i data-lucide="chevron-right" class="w-4 h-4 text-slate-300"></i>
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
                    fetchFiles();
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
                if (welcomeScreen) welcomeScreen.style.display = 'none';

                const div = document.createElement('div');
                div.className = `flex ${isAi ? 'justify-start' : 'justify-end'} msg-animate`;
                
                const sourceTag = source ? `<div class="mt-2 pt-2 border-t border-slate-100 text-[10px] font-bold text-indigo-500 uppercase tracking-tighter">${source} SEARCH</div>` : '';
                
                const content = `
                    <div class="max-w-[85%] ${isAi ? 'bg-white text-slate-800' : 'bg-indigo-600 text-white'} p-4 rounded-2xl shadow-sm border ${isAi ? 'border-slate-100' : 'border-indigo-500'}">
                        <div class="text-[15px] max-w-full overflow-x-auto leading-relaxed">${text.replace(/\\n/g, '<br>')}</div>
                        ${sourceTag}
                    </div>
                `;
                div.innerHTML = content;
                chatMessages.appendChild(div);
                const chatView = document.getElementById('chat-view');
                chatView.scrollTo({ top: chatView.scrollHeight, behavior: 'smooth' });
            }

            function showTyping() {
                const div = document.createElement('div');
                div.id = 'typing-indicator';
                div.className = 'flex justify-start msg-animate';
                div.innerHTML = `
                    <div class="bg-white p-4 rounded-2xl shadow-sm border border-slate-100 flex gap-1 items-center">
                        <span class="w-1.5 h-1.5 bg-indigo-400 rounded-full typing-dot"></span>
                        <span class="w-1.5 h-1.5 bg-indigo-400 rounded-full typing-dot"></span>
                        <span class="w-1.5 h-1.5 bg-indigo-400 rounded-full typing-dot"></span>
                    </div>
                `;
                chatMessages.appendChild(div);
                const chatView = document.getElementById('chat-view');
                chatView.scrollTo({ top: chatView.scrollHeight, behavior: 'smooth' });
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
                        body: JSON.stringify({ text, history: conversationHistory })
                    });
                    const data = await res.json();
                    document.getElementById('typing-indicator').remove();
                    addMessage(data.answer, true, data.source);
                    conversationHistory = data.history;
                    localStorage.setItem('lexai_history', JSON.stringify(conversationHistory));
                } catch (e) {
                    if (document.getElementById('typing-indicator')) {
                        document.getElementById('typing-indicator').remove();
                    }
                    addMessage("Technical glitch, yaar. Try again?", true);
                }
            }

            function handleKey(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    // Prevent default on mobile to prevent keyboard jumping
                    if (window.innerWidth > 768) {
                        e.preventDefault();
                        sendMessage();
                    }
                }
            }

            function newChat() {
                conversationHistory = [];
                localStorage.removeItem('lexai_history');
                chatMessages.innerHTML = '';
                if (welcomeScreen) welcomeScreen.style.display = 'block';
            }

            // Load existing history on start
            if (conversationHistory.length > 0) {
                conversationHistory.forEach(msg => addMessage(msg.content, msg.role === 'assistant'));
            } else {
                // Fetch localized greeting for new session
                fetch('/greeting')
                    .then(res => res.json())
                    .then(data => {
                        if (data.greeting) addMessage(data.greeting, true);
                    });
            }
        </script>
    </body>
    </html>
    """
