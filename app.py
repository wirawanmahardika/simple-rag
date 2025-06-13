from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import build
from query import getAnswer
import json
from database.db import database, metadata, engine
from database.crud import get_chats
from contextlib import asynccontextmanager

metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    print("✅ Database connected")

    yield

    await database.disconnect()
    print("❌ Database disconnected")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="templates")

@app.post("/store")
async def post_store(file: UploadFile = File(...), topic: str = Form(...)):
    if not file.filename:
        return JSONResponse({"error": "Nama file tidak valid"}, status_code=400)

    content = (await file.read()).decode("utf-8")
    docs = []
    if file.filename.endswith(".json"):
        data = json.loads(content)
        docs = [item["text"] for item in data if "text" in item]
    elif file.filename.endswith(".jsonl"):
        docs = []
        for line in content.splitlines():
            if line.strip():
                obj = json.loads(line)
                if "text" in obj:
                    docs.append(obj["text"])
    else:
        docs = [line.strip() for line in content.splitlines() if line.strip()]
    await build.runBuild(topic, docs)
    return JSONResponse({"message": "File berhasil diunggah dan dibaca"})

@app.get("/chats")
async def get_all_chats():
    return await get_chats()

@app.post("/search")
async def post_search(body: dict):
    try:
        question = body.get("question", "")
        topicName = body.get("topic", "")
        jawaban = getAnswer(question, name=topicName)
        return jawaban
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception:
        return JSONResponse({"error": "internal server error"}, status_code=500)
    
@app.get("/",response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})