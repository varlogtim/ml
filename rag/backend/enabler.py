import json
import logging
import requests

from .ingestor import (
    WebScraper,
    HtmlProcessor,
    TextProcessor,
    ChunkBatchProcessor,
    Html,
    Markdown,
    ChunkBatch,
    Metadata,
    VectorDbInput
)
from .vectordb import VectorDatabase

from pathlib import Path
from pydantic import AnyUrl as Url
from collections.abc import Generator, Iterable
from typing import Any

from chromadb.utils.embedding_functions import EmbeddingFunction


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndpointLLM:
    # TODO impl to_json, from_json for serialization
    def __init__(
        self, endpoint: Url | str, model_name: str, system_prompt: str,
        temperature: float | None, max_tokens: int | None
    ) -> None:
        self.endpoint = endpoint
        if isinstance(endpoint, str):
            self.endpoint = Url(endpoint)
        self.headers={"Content-Type": "application/json"}
        self.model_name = model_name
        self.system_prompt = system_prompt

        # TODO bounds check
        self.temperature = 0.7 if temperature is None else temperature
        self.max_tokens = 100 if max_tokens is None else max_tokens


    def __call__(self, user_query: str, temperature: float | None = None, max_tokens: int | None = None):
        # TODO validate within temperature bounds
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens if max_tokens else self.max_tokens
        }
        try:
            response = requests.post(self.endpoint, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            
            # TODO figure out exact structure and typehint return
            if "choices" in result and len(result["choices"]) > 0:
                assistant_message = result["choices"][0]["message"]["content"]
                return f"Assistant: {assistant_message}"
            else:
                return "No response received from the model."
        except requests.RequestException as e:
            raise RuntimeError(f"Error querying the endpoint: {e}")


# docker run -p 11433:80 --gpus all ghcr.io/huggingface/text-embeddings-inference:turing-0.6 --model-id sentence-transformers/all-MiniLM-L6-v2
class EndpointEmbedding(EmbeddingFunction):
    # TODO impl to_json, from_json for serialization
    def __init__(self, endpoint: Url | str):
        self.endpoint = endpoint
        if isinstance(endpoint, str):
            self.endpoint = Url(endpoint)
        self.headers={"Content-Type": "application/json"}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError(f"input must be a string or list of strings, not: {type(texts)}")

        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                data=json.dumps({"inputs": texts})
            )
            response.raise_for_status()
            resp_json = response.json()

            if isinstance(resp_json, list):
                ilen, olen = len(texts), len(resp_json)
                if ilen != olen:
                    raise ValueError(f"output len({olen}) not equal to input len({ilen})")
                return resp_json

            if isinstance(resp_json, dict):
                error = ret_obj.get("error")
                raise ValueError(f"error response from embedding server: {resp_json}")

            raise ValueError(f"unknown response from embedding server: {resp_json}")

        except requests.RequestException as e:
            raise RuntimeError(f"failed to get embeddings from {self.endpoint}: {e}")



ANSWER_SYSTEM_PROMPT = """
You are a helpful AI bot whose goal is to accellerate the gathering of
 information by a "user" related to technical issues of a product called ChromaDB.

You are provided lists of excerpts of previously reported product issues and documents.

The list of product issues excerpts will start with "#START_ISSUES" on a single line and
end the list with "#END_ISSUES" on a single line.

If you find a related issue match, please generate a URL to the issue and include it
 in your response. All of the issue urls should be sourced from 
 https://github.com/chroma-core/chroma/issues/

Include whether the issue was resolved by merging a PR. If a PR is located, please include
 the link to the PR as well.

The list of product issues excerpts will start with "#START_DOCS" on a single line and
end the list with "#END_DOCS" on a single line.

The query entered by the user will start with "#START_QUERY" and end with "#END_QUERY"
"""


ANSWER_USER_PROMPT = """
#START_ISSUES
{issues}
#END_ISSUES

#START_DOCS
{docs}
#END_DOCS

#START_QUERY
{query}
#END_QUERY
"""


class Enabler:
    def __init__(self, config: dict[str, Any]):
        self._load_config(config)
        self.embed_func = EndpointEmbedding(
            self.models_embed_endpoint,  # XXX Handle auth
        )
        self.answer_llm_func = EndpointLLM(
            endpoint=self.models_llm_endpoint,
            model_name=self.models_llm_name,
            system_prompt=ANSWER_SYSTEM_PROMPT,
            temperature=self.models_llm_temperature,
            max_tokens=self.models_llm_max_tokens
        )
        # TODO Think about a way to parameterize this.
        # .... when it's inside the container we can hardcode it.
        vectordb_data_path = "/home/ttucker/tmp/vector_db/"
        self.issues_vectordb = VectorDatabase(vectordb_data_path, "github_issues", self.embed_func)
        

    # TODO Tim, quit thinking about persistence and crap and get this working... wth.
    def _load_config(self, config: dict[str, Any]):
        # Top-level settings
        self.app_name: str = config.get("app_name", "The Enabler")  # Muhahahahaha....
        self.version: str = config.get("version", "0.6.9")
        self.debug_mode: bool = config.get("debug", False)
        self.max_retries: int = config.get("max_retries", 3)
        self.timeout_seconds: int = config.get("timeout_seconds", 10)

        # Models
        models_config = config.get("models", {})
        llm_config = models_config.get("llm", {})

        # LLM
        self.models_llm_name: str = llm_config.get("name", "llama3.1:latest")
        self.models_llm_endpoint: str = llm_config.get("endpoint", "http://localhost:11434/v1/chat/completions")
        self.models_llm_api_key: str = llm_config.get("api_key", None)
        self.models_llm_temperature: float = llm_config.get("temperature", 0.3)
        self.models_llm_max_tokens: int = llm_config.get("max_tokens", 2000)

        # Embedding
        embed_config = config.get("embedding", {})
        self.models_embed_name: str = embed_config.get("name", "sentence-transformers:all-MiniLM-L6-v2")
        self.models_embed_endpoint: str = embed_config.get("endpoint", "http://localhost:11433/")
        self.models_embed_api_key: str = embed_config.get("api_key", None)

        # XXX TODO XXX Think about how to set vectordb data storage path

    @classmethod
    def from_json(cls, json_str: str) -> 'AppConfig':
        try:
            config_dict = json.loads(json_str)
            if not isinstance(config_dict, dict):
                raise ValueError("JSON must represent a dictionary")
            return cls(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "version": self.version,
            "debug": self.debug_mode,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "models": {
                "llm": {
                    "name": self.models_llm_name,
                    "endpoint": self.models_llm_endpoint,
                    "api_key": self.models_llm_api_key,
                    "temperature": self.models_llm_temperature,
                    "max_tokens": self.models_llm_max_tokens
                },
                "embedding": {
                    "name": self.models_embed_name,
                    "endpoint": self.models_embed_endpoint,
                    "api_key": self.models_embed_api_key
                }
            }
        }

    def __str__(self) -> str:
        return self.to_json()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def query(self, text: str):
        # TODO Define a set of expected "query types"
        # .... first ask the LLM to determine which query type fits best.
        # .... then ask the LLM to distill the information from the request and fit
        # .... it into a template associated with that query type.
        # .... ^ literally, like defining fuzzy end points based on intention. 
        #
        # TODO think about possibly logging or updating a state that can be polled.

        # TODO pre process the input query with the LLM to condence query and make
        # .... it more applicable to each kind of data source we are querying.

        # Get VectorDB Results
        issue_results_raw = self.issues_vectordb.query(text)
        issue_results = "\n\n ##\n\n".join([
            result.encode().decode("unicode_escape") for result in issue_results_raw
        ])

        # Build Prompt
        answer_query = ANSWER_USER_PROMPT.format(
            issues=issue_results,
            docs="",
            query=text
        )

        return self.answer_llm_func(answer_query)
