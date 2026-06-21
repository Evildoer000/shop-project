from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes import router
from app.db.schema import ensure_database_schema
from app.db.session import get_engine
from app.services.organizer_dataset import OrganizerDatasetError, resolve_dataset_dir
from app.services.upload_storage import get_upload_dir


app = FastAPI(title="Ecommerce RAG Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.mount("/uploads", StaticFiles(directory=str(get_upload_dir())), name="uploads")


@app.on_event("startup")
def ensure_database_tables() -> None:
    ensure_database_schema(get_engine())

try:
    dataset_dir = resolve_dataset_dir()
except OrganizerDatasetError:
    dataset_dir = None
if dataset_dir is not None and Path(dataset_dir).exists():
    app.mount("/dataset", StaticFiles(directory=str(dataset_dir)), name="dataset")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
