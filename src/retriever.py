""" 
Hybrid Retrieval Module for Jarvis Mk.X
Uses: Dense embeddings + ChromaDB (dense) + BM25 (sparse) for hybrid retrieval.

Changes from v1:
- Embedding: all-MiniLM-L6-v2 (384-dim) → allenai-specter (768-dim, science-trained)
- Vector store: FAISS → ChromaDB (persistent, metadata-aware)
- Retrieval: Dense only → Hybrid (Dense + BM25 with score fusion)

Embedding backends:
- SentenceTransformers: model_name like "sentence-transformers/allenai-specter"
- Voyage API: model_name like "voyage-3-large" (requires VOYAGE_API_KEY)

Usage:
    from retriever import Retriever
    retriever = Retriever()
    retriever.build_index(paper.chunks)
    results = retriever.retrieve("What is attention?", top_k=5)
"""

import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import os


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (safe for zero vectors)."""
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array for normalization, got shape {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with `pip install sentence-transformers`."
            ) from e
        self._model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs,
    ):
        return self._model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


class _VoyageEmbedder:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        output_dimension: Optional[int] = None,
    ):
        try:
            import voyageai  # type: ignore
        except Exception as e:
            raise ImportError(
                "voyageai is required for Voyage embeddings. Install it with `pip install voyageai`."
            ) from e

        if api_key is None and not os.environ.get("VOYAGE_API_KEY"):
            raise ValueError(
                "VOYAGE_API_KEY is not set. Set the environment variable VOYAGE_API_KEY "
                "(or pass api_key=...) to use Voyage embeddings."
            )

        # voyageai.Client() will pick up VOYAGE_API_KEY automatically
        self._client = voyageai.Client(api_key=api_key) if api_key else voyageai.Client()
        self._model_name = model_name
        self._output_dimension = output_dimension
        self._embedding_dim: Optional[int] = None

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        input_type: Optional[str] = None,
        truncation: bool = True,
        **kwargs,
    ) -> np.ndarray:
        # Keep signature compatible with SentenceTransformer.encode; ignore show_progress_bar.
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i:i + batch_size]
            result = self._client.embed(
                batch,
                model=self._model_name,
                input_type=input_type,
                truncation=truncation,
                output_dimension=self._output_dimension,
            )
            all_embeddings.extend(result.embeddings)

        vectors = np.asarray(all_embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Voyage returned embeddings with unexpected shape")
        if self._embedding_dim is None and vectors.size:
            self._embedding_dim = int(vectors.shape[1])
        return _l2_normalize(vectors) if normalize_embeddings else vectors

    def get_sentence_embedding_dimension(self) -> int:
        if self._embedding_dim is not None:
            return self._embedding_dim
        # Lazily probe once (rarely called in this codebase).
        probe = self.encode(["dimension probe"], normalize_embeddings=False)
        self._embedding_dim = int(probe.shape[1])
        return self._embedding_dim


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""
    chunk_id: int
    text: str
    section: str
    page_numbers: List[int]
    score: float
    chunk_type: str = "text"
    retrieval_method: str = "hybrid"  # "dense", "sparse", or "hybrid"


class Retriever:
    """
    Hybrid retriever combining dense (SPECTER + ChromaDB) and sparse (BM25) search.

    Dense search captures semantic meaning ("How does the model work?" matches
    chunks about architecture even without exact word overlap).

    Sparse search captures exact keywords ("BLEU 28.4" matches chunks
    containing those exact terms).

    Hybrid fusion combines both for the best of both worlds.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/allenai-specter",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        voyage_api_key: Optional[str] = None,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            model_name: Sentence-transformers model for dense embeddings
            dense_weight: Weight for dense (semantic) scores in fusion (0-1)
            sparse_weight: Weight for sparse (BM25) scores in fusion (0-1)
        """
        self.model_name = model_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Load embedding model (local SentenceTransformers or Voyage API)
        if model_name.strip().lower().startswith("voyage-"):
            self.embed_model = _VoyageEmbedder(model_name=model_name, api_key=voyage_api_key)
        else:
            self.embed_model = _SentenceTransformerEmbedder(model_name)

        # Not currently used elsewhere, but handy for debugging.
        self.embedding_dim = None

        # ChromaDB client (in-memory for single-paper use)
        self.chroma_client = chromadb.Client()
        self.collection = None

        # BM25 index
        self.bm25 = None

        # Chunk storage
        self.chunks = None
        self.chunk_texts = None

    def build_index(self, chunks: list) -> None:
        """
        Build both dense (ChromaDB) and sparse (BM25) indices.

        Args:
            chunks: List of TextChunk objects from PaperProcessor
        """
        self.chunks = chunks
        self.chunk_texts = [chunk.text for chunk in chunks]

        # ─── Dense Index: ChromaDB with SPECTER embeddings ───

        # Delete old collection if it exists
        try:
            self.chroma_client.delete_collection("paper_chunks")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        # Generate embeddings
        embeddings = self.embed_model.encode(
            self.chunk_texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=len(self.chunk_texts) > 20,
            input_type="document" if self.model_name.strip().lower().startswith("voyage-") else None,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32).tolist()

        # Add to ChromaDB with metadata
        # IMPORTANT: when indexing multiple PDFs together, each PDF's chunks
        # typically start at chunk_id=0. If we use chunk.chunk_id for IDs,
        # Chroma will see duplicates (e.g., chunk_0, chunk_1, ...).
        # Use the *global* index within the combined list as the Chroma ID.
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chunk_id": i,  # global chunk index (matches ids)
                "source_chunk_id": chunk.chunk_id,  # original per-PDF chunk_id
                "section": chunk.section,
                "page_numbers": ",".join(map(str, chunk.page_numbers)),
                "chunk_type": chunk.chunk_type,
            }
            for i, chunk in enumerate(chunks)
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=self.chunk_texts,
            metadatas=metadatas,
        )

        # ─── Sparse Index: BM25 ───

        # Tokenize for BM25 (simple whitespace + lowercasing)
        tokenized_chunks = [self._tokenize_for_bm25(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Simple tokenization for BM25: lowercase, remove punctuation, split."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Hybrid retrieval: combine dense (ChromaDB) and sparse (BM25) results.

        Args:
            query: The user's question
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects sorted by hybrid score
        """
        if self.collection is None or self.chunks is None:
            raise ValueError("Index not built. Call build_index() first.")

        num_chunks = len(self.chunks)
        fetch_k = min(top_k * 2, num_chunks)  # Fetch more, then re-rank

        # ─── Dense retrieval (ChromaDB) ───
        query_embedding = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            input_type="query" if self.model_name.strip().lower().startswith("voyage-") else None,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()

        dense_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
        )

        # Build dense score map: chunk_id → score
        dense_scores = {}
        if dense_results and dense_results['ids'] and dense_results['ids'][0]:
            for idx, (doc_id, distance) in enumerate(
                zip(dense_results['ids'][0], dense_results['distances'][0])
            ):
                chunk_id = int(doc_id.split('_')[1])
                # ChromaDB cosine distance: lower = more similar
                # Convert to similarity: 1 - distance (for cosine space)
                similarity = 1.0 - distance
                dense_scores[chunk_id] = max(similarity, 0.0)

        # ─── Sparse retrieval (BM25) ───
        query_tokens = self._tokenize_for_bm25(query)
        bm25_scores_raw = self.bm25.get_scores(query_tokens)

        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
        sparse_scores = {
            i: score / max_bm25
            for i, score in enumerate(bm25_scores_raw)
        }

        # ─── Hybrid fusion ───
        all_chunk_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

        hybrid_scores = {}
        for chunk_id in all_chunk_ids:
            d_score = dense_scores.get(chunk_id, 0.0)
            s_score = sparse_scores.get(chunk_id, 0.0)
            hybrid_scores[chunk_id] = (
                self.dense_weight * d_score + self.sparse_weight * s_score
            )

        # Sort by hybrid score
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for chunk_id, score in ranked[:top_k]:
            if chunk_id >= len(self.chunks):
                continue
            chunk = self.chunks[chunk_id]

            # Determine which method contributed most
            d = dense_scores.get(chunk_id, 0.0)
            s = sparse_scores.get(chunk_id, 0.0)
            if d > 0 and s > 0:
                method = "hybrid"
            elif d > 0:
                method = "dense"
            else:
                method = "sparse"

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=chunk.text,
                section=chunk.section,
                page_numbers=chunk.page_numbers,
                score=float(score),
                chunk_type=chunk.chunk_type,
                retrieval_method=method,
            ))

        return results

    def get_context(self, query: str, top_k: int = 5, max_tokens: int = 1500) -> str:
        """
        Retrieve relevant chunks and format as context string for the LLM.

        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
            max_tokens: Maximum approximate tokens for combined context

        Returns:
            Formatted context string with source citations
        """
        results = self.retrieve(query, top_k=top_k)

        context_parts = []
        total_tokens = 0

        for result in results:
            approx_tokens = len(result.text) // 4
            if total_tokens + approx_tokens > max_tokens:
                break
            source_info = (
                f"[Section: {result.section} | "
                f"Page(s): {', '.join(map(str, result.page_numbers))} | "
                f"Match: {result.retrieval_method}]"
            )
            context_parts.append(f"{source_info}\n{result.text}")
            total_tokens += approx_tokens

        return "\n\n---\n\n".join(context_parts)
