from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer
from ai_vector_shield import VectorShield

class FastAPIVectorFirewall:
    def __init__(self, shield_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.shield = VectorShield.load(shield_path)
        self.model = SentenceTransformer(model_name)

    def verify_query(self, query: str) -> dict:
        emb = self.model.encode([query])
        result = self.shield.inspect(emb)[0]
        if not result["allowed"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Blocked by AIVectorShield: Out-of-distribution query or prompt injection detected.",
                    "risk_ratio": round(result["risk_ratio"], 4),
                    "chart_id": result["chart_id"],
                },
            )
        return result
