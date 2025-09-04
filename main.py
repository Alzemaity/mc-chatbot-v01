from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from rag_agent import RAGPipeline

app = FastAPI(title="MC LangGraph Chatbot")

rag = RAGPipeline()
rag.load_documents("data.json")  # ✅ load scraped data
rag.load_files("knowledge_base")  # ✅ load extra files

class ChatRequest(BaseModel):
    question: str

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.post("/chat")
async def chat(req: ChatRequest):
    answer = rag.answer(req.question)
    return {"answer": answer}
