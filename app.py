from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import build
from query import getAnswer
import json
from query import load_resources
from database.db import database, metadata, engine
from database.crud import get_chats
from contextlib import asynccontextmanager

metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await database.connect()
    print("✅ Database connected")

    yield  # <--- ini penting: app berjalan di antara startup & shutdown

    # Shutdown
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

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/input", response_class=HTMLResponse)
async def get_input(request: Request):
    return templates.TemplateResponse("input.html", {"request": request})

@app.post("/store")
async def post_store(file: UploadFile = File(...), nama: str = Form(...)):
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
    await build.runBuild(nama, docs)
    load_resources.cache_clear()
    return JSONResponse({"message": "File berhasil diunggah dan dibaca"})

@app.get("/chats")
async def get_all_chats():
    return await get_all_chats()

@app.post("/search")
async def post_search(body: dict):
    try:
        jawaban = getAnswer(body.get("question", ""))
        return jawaban
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception:
        return JSONResponse({"error": "internal server error"}, status_code=500)