#!/usr/bin/env python3
import json
import torch
import chromadb
import requests
import llama_index
import numpy as np

from pathlib import Path

# TODO remove
from typing import List, Union, Dict, Any, Generator

from collections.abc import Generator, Callable
from itertools import islice

from ingestor import VectorDbInput

# from langchain_community.document_loaders.base import BaseLoader
# from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

# Actually just going to use this: 
#  - https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339

# Parts:
# 1. Parsing
# 2. Embedding
# 3. UI?

#
# Common
#

class VectorDatabase:
    # XXX Need to worry about reingesting the same documents:
    def __init__(
            self, data_store: Path, collection_name: str, embed_func: Callable[[list[str]], list[list[float]]]
    ) -> None:
        """
        Some thing....

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


def main():
    vectordb_store = Path("./vector_store_data")
    vectordb_collection = "documents"
    embed = EndpointEmbedding("http://localhost:11433")  # XXX EMBEDDING_FUNC

    vector_db = VectorDatabase(vectordb_store, vectordb_collection, embed)

    document_loader = PyPDFDirectoryLoader("./tmp_documents_src")
    
    # XXX Skip loading documents for now
    vector_db.add_documents(document_loader)

    prompt = "What is a thread?"
    query_res = vector_db.query(prompt)
    
    # print(f"TTUCKER, get query response from Vector DB: {query_res}")

    PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

    # TODO rewrite this crap
    context_text = "\n\n - -\n\n".join(query_res)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm_prompt = prompt_template.format(context=context_text, question=prompt)
    
    #print(llm_prompt)

    print("TTUCKER: this is prompt: {llm_prompt}\n\n")

    llm = get_llm()
    response = llm.invoke(llm_prompt)

    print(response.content)

#main()
