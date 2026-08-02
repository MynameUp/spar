#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Descriptions] : PASA selector agent for relevance scoring
# ==================================================================

from typing import List
from log import logger


class Agent:
    """PASA (Paper-Aware Search Agent) selector for relevance scoring.

    Uses a fine-tuned LLM to evaluate paper relevance to queries.
    Default model: pasa-7b-selector
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        logger.info(f"PasaAgent initialized with model path: {model_path}")

    def _load_model(self):
        """Lazy load the PASA model."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                logger.info(f"Loaded PASA model from: {self.model_path}")
            except ImportError:
                logger.error(
                    "transformers/torch not installed. "
                    "Install with: pip install transformers torch"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load PASA model: {e}")
                raise

    def batch_infer_score(self, prompts: List[str], batch_size: int = 4) -> List[float]:
        """
        Batch inference for relevance scoring.

        Args:
            prompts: List of formatted prompt strings
            batch_size: Batch size for inference

        Returns:
            List of relevance scores (0-1 range)
        """
        self._load_model()

        scores = []
        try:
            import torch

            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(self._model.device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=0.0,
                    )

                for j, output in enumerate(outputs):
                    response = self._tokenizer.decode(
                        output[inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    score = self._parse_score(response)
                    scores.append(score)

            return scores

        except Exception as e:
            logger.error(f"Error in PASA batch inference: {e}")
            return [0.0] * len(prompts)

    def _parse_score(self, response: str) -> float:
        """
        Parse the relevance score from model response.
        Expected format: "Decision: True/False" or score value.

        Args:
            response: Model response string

        Returns:
            Parsed score (0.0 or 1.0)
        """
        import re

        response_lower = response.lower().strip()

        if "decision: true" in response_lower:
            return 1.0
        elif "decision: false" in response_lower:
            return 0.0

        # Try to extract a numeric score
        score_match = re.search(r"([0-1]\.\d+|\d+)", response)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))

        logger.warning(f"Could not parse score from response: {response[:100]}")
        return 0.0