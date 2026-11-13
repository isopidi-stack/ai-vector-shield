from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from ai_vector_shield import VectorShield
from typing import List

class SecureLangChainRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    shield: VectorShield
    model: SentenceTransformer

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_emb = self.model.encode([query])
        is_safe = self.shield.verify(query_emb)
        if not is_safe[0]:
            return [Document(page_content="[SECURITY ALERT]: Query blocked by AIVectorShield (OOD / Prompt Injection).")]
        return self.base_retriever.invoke(query, config={"callbacks": run_manager})