# app/main.py
"""
AI-Pulse Chatbot — UNRESTRICTED Backend
====================================
Request pipeline:
  1. Direct LLM agent call (UNRESTRICTED_SYSTEM_PROMPT + tools)
  2. If tool_call → Tavily search (Full Content) → synthesis LLM call
  3. Structured ChatResponse with FULL sources
"""

import logging
import os
from pathlib import Path
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from litellm import completion
from pydantic import BaseModel, Field
from .prompt_config import SYSTEM_PROMPT

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_pulse_unrestricted")

STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title="AI-Pulse UNRESTRICTED", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: list[dict] = Field(default_factory=list)

class SourceItem(BaseModel):
    title: str
    url: str
    full_content: str = ""

class ChatResponse(BaseModel):
    response: str
    status: str  
    sources: list[SourceItem] = []
    search_query: str = ""   

async def get_full_web_content(query: str) -> tuple[str, list[SourceItem]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY not configured.", []

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 5,
        "include_raw_content": True,
        "include_answer": True,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post("https://api.tavily.com/search", json=payload)
            r.raise_for_status()
            data = r.json()

        results = data.get("results", [])
        sources: list[SourceItem] = []
        context_parts: list[str] = []

        for res in results:
            title = res.get("title", "Source")
            url = res.get("url", "")
            raw_content = res.get("raw_content", res.get("content", ""))
            sources.append(SourceItem(title=title, url=url, full_content=raw_content))
            context_parts.append(f"SOURCE: {title}\nURL: {url}\nCONTENT: {raw_content}")

        return "\n\n---\n\n".join(context_parts), sources
    except Exception as e:
        logger.error(f"Search error: {e}")
        return "", []

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    user_query = request.query.strip()
    
    # إزالة كافة الفلاتر والقيود - الانتقال مباشرة إلى المعالجة
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in request.history[-10:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_query})

    try:
        # تحسين البحث لجلب المحتوى الكامل
        search_gen = completion(
            model=os.getenv("LLM_MODEL", "vertex_ai/gemini-2.0-flash-lite"),
            messages=[{"role": "system", "content": "Generate a precise search query for the user request. Output ONLY the query."},
                      {"role": "user", "content": user_query}],
            temperature=0.0
        )
        search_query = search_gen.choices[0].message.content.strip()
        
        full_context, sources = await get_full_web_content(search_query)
        
        # الرد النهائي بناءً على المحتوى الكامل وبدون قيود
        final_resp = completion(
            model=os.getenv("LLM_MODEL", "vertex_ai/gemini-2.0-flash-lite"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT FROM WEB:\n{full_context}\n\nUSER QUERY: {user_query}"}
            ],
            temperature=0.7
        )
        
        return ChatResponse(
            response=final_resp.choices[0].message.content,
            status="unrestricted_success",
            sources=sources,
            search_query=search_query
        )
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}", status="error")

