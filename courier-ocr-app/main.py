import os
import socket
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from src.api import router
from src.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Courier OCR", lifespan=lifespan)
app.include_router(router)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

if os.path.isdir(os.path.join(os.path.dirname(__file__), "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    ip = _local_ip()
    print(f"\n📱  Open on your phone: http://{ip}:{port}\n")
    uvicorn.run("main:app", host=host, port=port, reload=False)
