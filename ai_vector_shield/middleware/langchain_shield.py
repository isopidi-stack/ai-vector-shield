from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from ai_vector_shield import VectorShield
from typing import List

class VectorShieldBlockedError(PermissionError):
    """Raised when an inbound RAG query violates geometric manifold bounds."""

class SecureLangChainRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    shield: VectorShield
    model: SentenceTransformer
    filter_retrieved_docs: bool = True
    raise_on_block: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_emb = self.model.encode([query])
        if not self.shield.verify(query_emb)[0]:
            if self.raise_on_block:
                raise VectorShieldBlockedError("Inbound query blocked by AIVectorShield (OOD / Prompt Injection).")
            return []

        docs = self.base_retriever.invoke(query, config={"callbacks": run_manager})
        if self.filter_retrieved_docs and docs:
            doc_embs = self.model.encode([d.page_content for d in docs])
            safe_mask = self.shield.verify(doc_embs)
            docs = [d for d, ok in zip(docs, safe_mask) if ok]
        return docs
