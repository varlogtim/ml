#!/usr/bin/env python3
import chromadb

from .ingestor import VectorDbInput

from pathlib import Path
from collections.abc import Generator, Callable


class VectorDatabase:
    # XXX Need to worry about reingesting the same documents:
    def __init__(
            self, data_store: Path, collection_name: str, embed_func: Callable[[list[str]], list[list[float]]]
    ) -> None:
        """
        Something....

        In the docs, when the collection is created, the embedding function is linked
        to that collection. (TODO: find the docs on this again)

        """

        # TODO make dir if not exist?
        self.data_store = data_store
        self.collection_name = collection_name
        self.settings = chromadb.get_settings()
        self.settings.allow_reset = True
        self.failures: list[dict[str, str]] = []

        self.client = chromadb.PersistentClient(
            path=str(self.data_store),  # XXX wild that this doesn't accept a Path
            settings=self.settings
        )
        # XXX I couldn't use LangChain Chroma because they want some bunk other embedding function.

        # TODO remove this deletion after done testing
        # try:
        #     self.client.delete_collection(name=self.collection_name)
        # except chromadb.errors.NotFoundError:
        #     pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # XXX If modify, change relevance function
            embedding_function=embed_func,
        )

    def reset(self):
        self.client.reset()

    def add(self, vectordb_input: VectorDbInput):
        self.collection.add(
            documents=vectordb_input.texts,
            embeddings=vectordb_input.embeddings,
            metadatas=vectordb_input.metadatas,
            ids=vectordb_input.ids
        )

    # TODO support a list of prompts
    def query(self, prompt: str) -> list:
        relevance_score_cutoff = 0.3
        num_results = 10

        results = self.collection.query(
            query_texts=[prompt],
            n_results=num_results,
            # where={"metadata_field": "is_equal_to_this"},
            # where_document={"$contains":"search_string"},
        )
        # results = {
        #  "ids": [[p1_id1, p1_id2, ...], [p2_id1, p2_id2, ...], ...]
        #  "embeddings": [[p1_emb1, p1_emb2, ...], [p2_emb1, p2_emb2, ...], ...]
        #  "documents": [[p1_doc1, p1_doc2, ...], [p2_doc1, p2_doc2, ...], ...]
        #  "uris": [[...], ...]
        #  "included": [[...], ...]
        #  "data": [[...], ...]
        #  "metadatas": [[...], ...]
        #  "distances": [[...], ...]  # XXX these are cosine distances

        def get_relevance(distance: float) -> float:
            return 1 - (distance / 2.0)  # Normalize to [0, 1] for cosine distance 

        # TODO when list of prompts, change this algo
        # TODO this name sucks 
        doc_dists = zip(results["documents"][0], results["distances"][0])
        # TODO Change this...

        # evaluated_text = text.encode().decode('unicode_escape')

        filtered_results = [
            dd[0] for dd in doc_dists
            if (1 - (dd[1] / 2.0)) >= relevance_score_cutoff
        ]
        if len(filtered_results) == 0:
            print(f"Unable to find matching results.")

        return filtered_results
