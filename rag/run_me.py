import json
import logging
import requests

from pathlib import Path
from ingestor import (
    WebScraper, HtmlProcessor, TextProcessor, ChunkBatchProcessor,
    Html, Markdown, ChunkBatch, Metadata, VectorDbInput
)
from vectordb import VectorDatabase
from pydantic import AnyUrl as Url
from collections.abc import Generator, Iterable

from chromadb.utils.embedding_functions import EmbeddingFunction

# https://github.com/chroma-core/chroma/issues?q=is%3Aissue%20state%3Aclosed%20sort%3Acreated-desc
# https://github.com/chroma-core/chroma/issues/4643
# ^ this gives us the URL to the last issue.

# If there is a pull here: 
# https://github.com/chroma-core/chroma/pull/4649
# ... and that is in the comments, then we should address sentiment.

# XXX this one has a cool error in it.
# less da69cc60af42bb5ccc0c725f3cff4404.markdown

# XXX Can get issue url from markdown using:
# [ Sign in ](/login?return_to=https%3A%2F%2Fgithub.com%2Fchroma-core%2Fchroma%2Fissues%2F1521)


CACHE_PATH = Path("/home/ttucker/tmp/webscraper_cache")
VECTOR_PATH = Path("/home/ttucker/tmp/vector_db")

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

        # TODO bound check
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
                print(f"Assistant: {assistant_message}")
            else:
                print("No response received from the model.")
        except requests.RequestException as e:
            print(f"Error querying the endpoint: {e}")


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


#
# Pipeline functions:
#
def issues_pipline(embed_func: EmbeddingFunction, issues_vectordb: VectorDatabase):
    # TODO You can actually move the generators inside of ingestor.py
    # .... and just log the errors there. You only use these wrappers for logging.
    def IssueUrls() -> Generator[Url, None, None]:
        last_issue_id = 4643
        base_url = "https://github.com/chroma-core/chroma/issues/"

        for ii in range(1, last_issue_id + 1):
            yield Url(f"{base_url}{ii}")


    def Htmls(
        scraper: WebScraper, urls: Iterable[Url]
    ) -> Generator[tuple[Url, Html], None, None]:

        for url, html in scraper.get_all(urls):
            logger.info(f"received html: {url}, data len: {len(html)}")
            yield url, html

        failures = scraper.get_failures()
        logger.info(f"number of web scraper failures: {len(failures)}")
        for failure in failures:
            logger.error(f"web scraper failure: {failure}")


    def Markdowns(
        html_p: HtmlProcessor, urls_htmls: Iterable[tuple[Url, Html]]
    ) -> Generator[tuple[Url, Markdown], None, None]:

        for url, markdown in html_p.process_all(urls_htmls):
            logger.info(f"received markdown: {url}, data len: {len(markdown)}")
            yield url, markdown

        failures = html_p.get_failures()
        logger.info(f"number of html processor failures: {len(failures)}")
        for failure in failures:
            logger.error(f"html processor failure: {failure}")


    def ChunkBatches(
        text_p: TextProcessor, urls_mkdns: Iterable[tuple[Url, Markdown]]
    ) -> Generator[tuple[Url, ChunkBatch], None, None]:
        for url, chunk_batch in text_p.process(urls_mkdns):
            logger.info(f"received chunk batch: {url}, len(batch): {len(chunk_batch)}")
            yield url, chunk_batch

        failures = text_p.get_failures()
        logger.info(f"number of text processor failures: {len(failures)}")
        for failure in failures:
            logger.error(f"text processor failure: {failure}")

    def VectorDbInputs(
        ckbt_p: ChunkBatchProcessor, keys_ckbts: Iterable[tuple[str, ChunkBatch]]
    ) -> Generator[tuple[str, VectorDbInput], None, None]:
        for key, vectordb_input in ckbt_p.process(keys_ckbts):
            logger.info(f"received vectordb input: {key}")
            yield key, vectordb_input

        failures = ckbt_p.get_failures()
        logger.info(f"number of chunk batch processor failures: {len(failures)}")
        for failure in failures:
            logger.error(f"chunk batch processor failure: {failure}")

    def VectorDbInsert(
        vector_db: VectorDatabase, keys_vctrs: Iterable[tuple[str, VectorDbInput]]
    ) -> None:
        for key, vectordb_input in keys_vctrs:
            try:
                vector_db.add(vectordb_input)
                logger.info(f"inserted into collection: {vector_db.collection_name} data for {key}")
            except Exception as e:
                logger.error(f"failed to insert data for {key} into collection {vector_db.collection_name}")


    scraper = WebScraper(CACHE_PATH)
    html_p = HtmlProcessor(CACHE_PATH)
    text_p = TextProcessor(batch_size=32, chunk_size=512, chunk_overlap=100)
    ckbt_p = ChunkBatchProcessor(CACHE_PATH, embed_func)
    # docker run -p 11433:80 --gpus all ghcr.io/huggingface/text-embeddings-inference:turing-0.6 --model-id sentence-transformers/all-MiniLM-L6-v2

    issue_urls = IssueUrls()
    # subset_urls = []
    # for _ in range(100):
    #     subset_urls.append(next(issue_urls))
    # urls_htmls = Htmls(scraper, subset_urls)

    urls_htmls = Htmls(scraper, issue_urls)
    urls_mkdns = Markdowns(html_p, urls_htmls)
    urls_ckbts = ChunkBatches(text_p, urls_mkdns)
    urls_vctrs = VectorDbInputs(ckbt_p, urls_ckbts)

    VectorDbInsert(issues_vectordb, urls_vctrs)


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

# Ohhh... This query is so juicy...
# test_query = (
#     "I am getting the error, "
#     "'Could not connect to tenant default_tenant. Are you sure it exists?'"
#     "Please help!?"
# )



def main():
    embed_func = EndpointEmbedding("http://localhost:11433")
    answer_llm_func = EndpointLLM(
        endpoint="http://localhost:11434/v1/chat/completions",
        model_name="llama3.1:latest",
        system_prompt=ANSWER_SYSTEM_PROMPT,
        temperature=0.5,
        max_tokens=500
    )

    issues_vectordb = VectorDatabase(VECTOR_PATH, "github_issues", embed_func)

    test_query = (
        "I am getting the error, "
        "'Could not connect to tenant default_tenant. Are you sure it exists?'"
        "Please help!?"
    )

    process = False
    if process:
        issues_pipline(embed_func, issues_vectordb)

    # Get VectorDB Results
    issue_results_raw = issues_vectordb.query(test_query)
    issue_results = "\n\n ##\n\n".join([
        result.encode().decode("unicode_escape") for result in issue_results_raw
    ])

    # Build Prompt
    answer_query = ANSWER_USER_PROMPT.format(
        issues=issue_results,
        docs="",
        query=test_query
    )

    answer_llm_func(answer_query)

    # Send this to LLM


main()
