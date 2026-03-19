"""Retriever wrappers for LATTICE extensions."""

from __future__ import annotations

from retrievers.diver import DiverEmbeddingModel
from retrievers.reasonembed import ReasonEmbedEmbeddingModel

_REASONEMBED_TASK_DESCRIPTION_BY_SUBSET = {
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "psychology": "Given a Psychology post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
    "stackoverflow": "Given a Stack Overflow post, retrieve relevant passages that help answer the post.",
    "sustainable_living": "Given a Sustainable Living post, retrieve relevant passages that help answer the post.",
    "leetcode": "Given a Coding problem, retrieve relevant examples that help answer the problem.",
    "pony": "Given a Pony question, retrieve relevant passages that help answer the question.",
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoq": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theot": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
}


def is_reasonembed_model_path(model_path: str) -> bool:
    value = str(model_path or "").strip().lower()
    return ("reasonembed" in value) or ("reason-embed" in value)


def get_reasonembed_task_description(subset: str) -> str:
    subset_name = str(subset or "").strip().lower()
    if subset_name not in _REASONEMBED_TASK_DESCRIPTION_BY_SUBSET:
        raise ValueError(f"Unsupported subset for ReasonEmbed task instruction: {subset}")
    return _REASONEMBED_TASK_DESCRIPTION_BY_SUBSET[subset_name]


def build_retriever(
    model_path: str,
    *,
    subset: str | None = None,
    local_files_only: bool = True,
    **kwargs,
):
    if is_reasonembed_model_path(model_path):
        if subset is None:
            raise ValueError("subset is required when building a ReasonEmbed retriever")
        task_description = get_reasonembed_task_description(subset)
        # Intent: keep reason-embed query encoding aligned with the subset-specific instruction used by the model.
        return ReasonEmbedEmbeddingModel(
            model_path,
            local_files_only=local_files_only,
            task_description=task_description,
            **kwargs,
        )
    return DiverEmbeddingModel(model_path, local_files_only=local_files_only, **kwargs)


__all__ = [
    "DiverEmbeddingModel",
    "ReasonEmbedEmbeddingModel",
    "build_retriever",
    "get_reasonembed_task_description",
    "is_reasonembed_model_path",
]
