# ============================================
# ChromaDB Vector Database Module (LangChain)
# ============================================
# Manages the Chroma vector store used for
# Retrieval-Augmented Generation (RAG).
#
# Fully implemented using LangChain components:
#   • HuggingFaceBgeEmbeddings from langchain_community
#   • Chroma vector store wrapper from langchain_community
#   • RecursiveCharacterTextSplitter for document chunking
#   • Document schema from langchain_core
#
# Responsibilities:
#   • Initialize persistent Chroma vector store
#   • Seed the DB with HumanEval dataset
#   • Perform similarity search (top-k retrieval)
#   • Insert new documents from user feedback
#   • Expose LangChain retriever interface
# ============================================

import os
from typing import Optional, List, Dict, Any
from datasets import load_dataset

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    TOP_K_RESULTS,
    HUMANEVAL_DATASET,
    HUMANEVAL_SPLIT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class VectorDatabase:

    def __init__(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # 1. Initialize LangChain embedding model
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
        )

        # 2. Initialize LangChain persistent Chroma vector store
        self.vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding,
            persist_directory=CHROMA_PERSIST_DIR,
        )

        # 3. Initialize LangChain code text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\ndef ", "\nclass ", "\n\n", "\n", " ", ""],
        )

    def count(self) -> int:
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    # ──────────────────────────────────────────
    # Dataset Seeding
    # ──────────────────────────────────────────

    def seed_from_humaneval(self) -> int:
        current_count = self.count()
        if current_count > 0:
            print(
                f"[VDB] Collection already contains {current_count} documents. Skipping seed."
            )
            return current_count

        print("[VDB] Loading HumanEval dataset via HuggingFaceDatasetLoader...")
        loader = HuggingFaceDatasetLoader(
            path="openai/openai_humaneval",
            page_content_column="prompt",
            name="openai_humaneval"
        )
        raw_documents = loader.load()

        for idx, doc in enumerate(raw_documents):
            doc.metadata.setdefault("source", "HumanEval")
            doc.metadata.setdefault("index", idx)

        print(f"[VDB] Splitting {len(raw_documents)} HumanEval documents with LangChain text splitter...")
        split_docs = self.text_splitter.split_documents(raw_documents)

        # Assign unique IDs
        ids = [f"humaneval_{i}" for i in range(len(split_docs))]

        print(f"[VDB] Indexing {len(split_docs)} chunks into ChromaDB...")
        batch_size = 100
        for i in range(0, len(split_docs), batch_size):
            end = min(i + batch_size, len(split_docs))
            self.vector_store.add_documents(
                documents=split_docs[i:end],
                ids=ids[i:end],
            )

        new_count = self.count()
        print(f"[VDB] Seeded {new_count} document chunks into ChromaDB.")
        return new_count

    # ──────────────────────────────────────────
    # Similarity Search
    # ──────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = TOP_K_RESULTS

        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        formatted = []
        for doc, score in results:
            formatted.append(
                {
                    "document": doc.page_content,
                    "metadata": doc.metadata,
                    "distance": float(score),
                }
            )

        return formatted

    def as_retriever(self, **kwargs):
        search_kwargs = kwargs.pop("search_kwargs", {"k": TOP_K_RESULTS})
        return self.vector_store.as_retriever(search_kwargs=search_kwargs, **kwargs)

    # ──────────────────────────────────────────
    # User Feedback Insertion
    # ──────────────────────────────────────────

    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if metadata is None:
            metadata = {}

        current_count = self.count()
        doc_id = f"user_feedback_{current_count}"
        metadata.setdefault("source", "user_feedback")

        base_doc = Document(page_content=content, metadata=metadata)
        chunks = self.text_splitter.split_documents([base_doc])

        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        self.vector_store.add_documents(documents=chunks, ids=chunk_ids)

        print(f"[VDB] Added new document: {doc_id} ({len(chunks)} chunks)")
        return doc_id

    # ──────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": CHROMA_COLLECTION_NAME,
            "document_count": self.count(),
        }
