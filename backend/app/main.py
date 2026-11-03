from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .analyzer import analyze
from .bert_models import models
from .config import settings
from .schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading BERT models on {models.device.upper()}" + (f" ({models.gpu_name})" if models.gpu_name else "") + "...")
    models.load()
    print(f"Models ready. LLM check: {'on (' + settings.bedrock_model_id + ')' if settings.llm_configured else 'off - set BEDROCK_MODEL_ID and AWS credentials in .env'}")
    yield


app = FastAPI(title="Comment Guard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if models.loaded else "loading",
        device=models.device,
        gpu_name=models.gpu_name,
        llm_configured=settings.llm_configured,
        llm_model=settings.bedrock_model_id,
    )


# A plain `def` endpoint: FastAPI runs it in a worker thread, so model inference doesn't block the server.
@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_comment(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Comment is empty.")
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models are still loading - try again in a moment.")
    return analyze(request.text.strip(), use_llm=request.use_llm)


# After `npm run build`, the backend serves the page too, so the whole app runs on one port.
# Mounted last so the /api routes above take precedence.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
