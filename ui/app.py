from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import load_rag_engine, ask

app = FastAPI()
engine = None

class Question(BaseModel):
    text: str

@app.on_event("startup")
async def startup():
    global engine
    print("Loading RAG engine...")
    engine = load_rag_engine()
    print("Ready.")

@app.post("/ask")
async def ask_question(q: Question):
    answer = ask(engine, q.text)
    return {"answer": answer}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html><body>
    <h2>LexAI</h2>
    <input id="q" placeholder="Ask something..." style="width:400px">
    <button onclick="send()">Ask</button>
    <p id="answer"></p>
    <script>
    async function send() {
        const btn = document.querySelector('button');
        btn.disabled = true;
        btn.innerText = 'Thinking...';
        const res = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: document.getElementById('q').value})
        });
        const data = await res.json();
        document.getElementById('answer').innerText = data.answer;
        btn.disabled = false;
        btn.innerText = 'Ask';
    }
    </script>
    </body></html>
    """