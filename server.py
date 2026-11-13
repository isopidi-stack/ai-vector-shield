import os
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from ai_vector_shield import VectorShield

app = FastAPI(
    title="AI Vector Shield Enterprise Sidecar",
    description="Sub-millisecond geometric RAG firewall powered by piecewise-local CP^d manifolds.",
    version="0.1.0",
)

SHIELD_PATH = os.getenv("VECTOR_SHIELD_MODEL_PATH", "")
DEFAULT_MODE = os.getenv("VECTOR_SHIELD_MODE", "enforce")
DEFAULT_DIM = int(os.getenv("VECTOR_SHIELD_BOOTSTRAP_DIM", "384"))

shield: Optional[VectorShield] = None


class FitRequest(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Trusted corpus embeddings [N, D]")
    latent_rank: int = 8
    n_clusters: int = 3
    quantile: float = 0.99
    mode: str = "enforce"


class InspectRequest(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Query or chunk embeddings [B, D]")


@app.on_event("startup")
def startup_event():
    global shield
    if SHIELD_PATH and os.path.exists(SHIELD_PATH):
        shield = VectorShield.load(SHIELD_PATH, mode=DEFAULT_MODE)
    else:
        # Bootstrap with a synthetic manifold so /healthz and trial calls work out-of-the-box
        rng = np.random.default_rng(42)
        bootstrap_emb = rng.normal(size=(120, DEFAULT_DIM)).astype(np.float32)
        shield = VectorShield(latent_rank=8, n_clusters=3, mode=DEFAULT_MODE).fit(bootstrap_emb)


@app.get("/healthz")
def healthz():
    return {
        "status": "healthy",
        "fitted": shield is not None and shield._is_fitted,
        "mode": shield.mode if shield else None,
        "dim": shield._shield.dim if (shield and shield._shield) else None,
    }


@app.post("/v1/fit")
def fit_manifold(req: FitRequest):
    global shield
    arr = np.asarray(req.embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a 2D array with at least 6 embedding rows.",
        )
    shield = VectorShield(
        latent_rank=req.latent_rank,
        n_clusters=req.n_clusters,
        quantile=req.quantile,
        mode=req.mode,
    ).fit(arr)
    if SHIELD_PATH:
        shield.save(SHIELD_PATH)
    return {"status": "fitted", "samples": int(arr.shape[0]), "dim": int(arr.shape[1]), "mode": shield.mode}


@app.post("/v1/inspect")
def inspect_embeddings(req: InspectRequest):
    if shield is None or not shield._is_fitted:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Shield not fitted.")
    try:
        arr = np.asarray(req.embeddings, dtype=np.float32)
        diagnostics = shield.inspect(arr)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"results": diagnostics}


@app.post("/v1/verify")
def verify_embeddings(req: InspectRequest):
    if shield is None or not shield._is_fitted:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Shield not fitted.")
    try:
        arr = np.asarray(req.embeddings, dtype=np.float32)
        allowed = shield.verify(arr).tolist()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"allowed": allowed}
