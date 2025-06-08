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


# XXX NOT WORKING DUE TO PARTIAL REFACTOR !!!
# XXX NOT WORKING DUE TO PARTIAL REFACTOR !!!
# XXX NOT WORKING DUE TO PARTIAL REFACTOR !!!
# XXX NOT WORKING DUE TO PARTIAL REFACTOR !!!
# XXX NOT WORKING DUE TO PARTIAL REFACTOR !!!

# XXX TODO XXX Need to pass these in somehow
CACHE_PATH = Path("/home/ttucker/tmp/webscraper_cache")
VECTOR_PATH = Path("/home/ttucker/tmp/vector_db")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    urls_htmls = Htmls(scraper, issue_urls)
    urls_mkdns = Markdowns(html_p, urls_htmls)
    urls_ckbts = ChunkBatches(text_p, urls_mkdns)
    urls_vctrs = VectorDbInputs(ckbt_p, urls_ckbts)


def main():
    # XXX TODO XXX need to import this from somewhere.
    embed_func = EndpointEmbedding("http://localhost:11433")
    issues_vectordb = VectorDatabase(VECTOR_PATH, "github_issues", embed_func)

    process = False
    if process:
        issues_pipline(embed_func, issues_vectordb)

main()
