#!/usr/bin/env python3
import json
import torch
import chromadb
import requests
import logging
# import numpy as np

from pathlib import Path
from typing import List, Union, Dict, Any, Generator
from itertools import islice

from chromadb.utils.embedding_functions import EmbeddingFunction

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Parts:
# 1. Parsing
# 2. Embedding
# 3. UI?

def get_llm(endpoint: str):
    # TODO I don't think I need this stupid ChatOpenAI class...
    # TODO validate URL
    llm = ChatOpenAI(
        openai_api_base=f"{endpoint}/v1",
        openai_api_key="XXX_NOOP",
        model_name="llama3.1:latest"
    )
    # Query:
    # from langchain.schema import HumanMessage
    # messages = [HumanMessage(content="Hello, what is the capital of France?")]
    # response = llm.invoke(messages)

    return llm


class EndpointEmbedding(EmbeddingFunction):
    def __init__(self, endpoint: str):
        # TODO Need to implement auth
        # TODO Need to validate endpoint url
        self.endpoint = endpoint 
        self.headers={"Content-Type": "application/json"}
        return

    def __call__(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("Input must be a string or list of strings")

        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                data=json.dumps({"inputs": texts})
            )
            response.raise_for_status()
            ret_obj = response.json()

            if isinstance(ret_obj, list):
                # TODO, maybe validate the responses a bit more?
                return ret_obj

            if isinstance(ret_obj, dict):
                error = ret_obj.get("error")
                raise ValueError(f"Got an error response from embedding server: {ret_obj}")

            raise ValueError(f"Got an unknown response from embedding server: {ret_obj}")



        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get embeddings from {self.endpoint}: {e}")


class VectorDatabaseInterface:
    # TODO Impl destructors to use as context for cleanup
    # TODO Need to worry about reingesting the same documents:
    def __init__(self, data_store: Path, collection_name: str, embed_func: EmbeddingFunction):
        """
        Talk to the VectorDB

        In the docs, when the collection is created, the embedding function is linked
        to that collection. (TODO: find the docs on this again)

        """

        # TODO make dir if not exist?
        self.data_store = data_store
        self.collection_name = collection_name
        self.embed_func = embed_func

        self.settings = chromadb.get_settings()
        self.settings.allow_reset = True

        self.client = chromadb.PersistentClient(
            path=str(self.data_store),  # XXX wild that this doesn't accept a Path
            settings=self.settings
        )

        self.__init_collection()

    def __init_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # XXX If modify, change relevance function
            embedding_function=self.embed_func,
        )

    def reset(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except chromadb.errors.NotFoundError:
            pass

        self.__init_collection()

        #self.client.reset()


    # TODO hint types in Generator return
    def _document_batches(self, document_loader: BaseLoader) -> Generator:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, # Size of each chunk in characters
            chunk_overlap=100, # Overlap between consecutive chunks
            length_function=len, # Function to compute the length of the text
            add_start_index=True, # Flag to add start index to each chunk
        )

        doc_batch_size = 1024  # XXX this may adjust Python memory usage. 
        def batched_docs(document_loader: BaseLoader, doc_batch_size: int) -> Generator:
            loader = document_loader.lazy_load()
            while True:
                doc_batch = list(islice(loader, doc_batch_size))
                if not doc_batch:
                    break
                yield doc_batch

        # TODO, after you understand, change the name of this.
        batch_size = 32  # XXX EMBEDDING_FUNC limitation - not sure how to set this.
        batch = []
        for docs_batch in batched_docs(document_loader, doc_batch_size):
            for chunk_idx, chunk in enumerate(text_splitter.split_documents(docs_batch)):
                doc_id = f"{chunk.metadata['source']}-{chunk_idx}"
                batch.append((chunk.page_content, chunk.metadata, doc_id))

                if (len(batch) + 1) % (batch_size + 1) == 0:
                    docs, metas, ids = zip(*batch)
                    yield docs, metas, ids
                    batch = []

        if len(batch):
            docs, metas, ids = zip(*batch)
            yield docs, metas, ids


    def add_documents(self, document_loader: BaseLoader):
        for docs, metas, ids in self._document_batches(document_loader):
            self.collection.add(documents=list(docs), metadatas=list(metas), ids=list(ids))

    # TODO support a list of prompts
    def query(self, prompt: str) -> list:
        relevance_score_cutoff = 0.7
        num_results = 10

        results = self.collection.query(
            query_texts=[prompt],
            n_results=num_results,
            # where={"metadata_field": "is_equal_to_this"},
            # where_document={"$contains":"search_string"},
        )
        # Struct:
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

        filtered_results = [
            dd[0] for dd in doc_dists
            if (1 - (dd[1] / 2.0)) >= relevance_score_cutoff
        ]
        if len(filtered_results) == 0:
            print(f"Unable to find matching results.")

        return filtered_results



# TODO Remove all these globals
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

# TODO Impl a Url() class wrapper for these URLs. Or look for one. There probably is one somewhere.

# XXX EMBEDDING_FUNC
# docker run -p 11433:80 --gpus all ghcr.io/huggingface/text-embeddings-inference:turing-0.6 --model-id sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_ENDPOINT = "http://localhost:11433"

# XXX LLM_FUNC
# TODO Running against ollama, change to docker container.
LLM_ENDPOINT = "http://localhost:11434"

VECTORDB_STORE = Path("/tmp/vector_store")

# TODO implement some better directory searcher.
DOCUMENT_DIR = Path("/tmp/documents")

VECTORDB_COLLECTION_NAME = "documents"


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("START")
    embed: EmbeddingFunction = EndpointEmbedding(EMBEDDING_ENDPOINT)  # XXX EMBEDDING_FUNC

    vectordb = VectorDatabaseInterface(VECTORDB_STORE, VECTORDB_COLLECTION_NAME, embed)
    logger.info("VectorDB initialized")

    document_loader = PyPDFDirectoryLoader(DOCUMENT_DIR)
    logger.info("PyPDF Dir loader initialized")
    
    vectordb.reset()
    logger.info("VectorDB reset")
    vectordb.add_documents(document_loader)
    logger.info("Added documents")

    prompt = "What is a thread?"
    vectordb_response = vectordb.query(prompt)
    logger.info("Received vectorDB query response")


    # TODO rewrite this crap
    context_text = "\n\n - -\n\n".join(vectordb_response)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    print(f"TTUCKER: type(prompt_template): {type(prompt_template)}")
    llm_prompt = prompt_template.format(context=context_text, question=prompt)
    
    llm = get_llm(LLM_ENDPOINT)
    response = llm.invoke(llm_prompt)
    logger.info("Got LLM Response")

    print(f"Question: {prompt}\n\nRAG Response: {response.content}")

main()
