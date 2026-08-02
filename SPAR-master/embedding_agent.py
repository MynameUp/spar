#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Descriptions] : BGE-M3 embedding agent for similarity scoring
# ==================================================================

import numpy as np
from typing import List
from log import logger


class BGEM3EmbeddingAgent:
    """BGE-M3 embedding model agent for computing similarity scores.

    Uses BAAI/bge-m3 model for dense embedding-based similarity computation.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None
        logger.info(f"BGEM3EmbeddingAgent initialized with model: {model_name}")

    def _load_model(self):
        """Lazy load the BGE-M3 model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded BGE-M3 model: {self.model_name}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load BGE-M3 model: {e}")
                raise

    def get_score(self, query: str, documents: List[str], batch_size: int = 6) -> List[float]:
        """
        Compute similarity scores between a query and a list of documents.

        Args:
            query: The query string
            documents: List of document text representations
            batch_size: Batch size for encoding

        Returns:
            List of similarity scores (0-1 range)
        """
        self._load_model()

        try:
            # Encode query and documents
            query_embedding = self._model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )

            doc_embeddings = self._model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )

            # Compute cosine similarity
            scores = np.dot(doc_embeddings, query_embedding.T).flatten()
            # Clamp to [0, 1]
            scores = np.clip(scores, 0, 1)

            return scores.tolist()

        except Exception as e:
            logger.error(f"Error computing BGE-M3 scores: {e}")
            # Return default low scores on failure
            return [0.0] * len(documents)

    def cleanup(self):
        """Release model resources."""
        self._model = None