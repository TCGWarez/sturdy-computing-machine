"""CLIP embedding module"""
from .embedder import CLIPEmbedder, get_embedder, get_image_embedding, get_batch_embeddings

__all__ = ['CLIPEmbedder', 'get_embedder', 'get_image_embedding', 'get_batch_embeddings']