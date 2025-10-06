"""Tag tree utilities exposed by the rmsearch package."""

from .generate_tag import build_pool_from_settings, generate_tag
from .embed_tags import embed_pool_context, embed_tags
from .hierarchical_kmeans import HierarchicalKMeans
from .build_representative_tags import (
    build_representative_tags,
    extract_text,
    get_node_by_path,
    get_representative_tag_request,
    is_leaf,
    set_representative_tag,
)

__all__ = [
    "build_pool_from_settings",
    "generate_tag",
    "embed_pool_context",
    "embed_tags",
    "HierarchicalKMeans",
    "build_representative_tags",
    "extract_text",
    "get_node_by_path",
    "get_representative_tag_request",
    "is_leaf",
    "set_representative_tag",
]
