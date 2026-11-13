from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer
from ai_vector_shield import VectorShield
import torch

class FastAPIVectorFirewall:
    def __init__(self, shield_path: str, model_name: str = "all-MiniLM-L6-v2"):
        print("[*] Loading VectorFirewall and embedding model for FastAPI...")
        self.shield = VectorShield.load(shield_path)
        self.model = SentenceTransformer(model_name)

    def verify_query(self, query: str):
        emb = self.model.encode([query])
        is_safe = self.shield.verify(emb)
        if not is_safe[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request blocked by AIVectorShield: Out-of-distribution data or potential prompt injection detected."
            )
        return True