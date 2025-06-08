import os
import hashlib
import time
import json
import requests
import html2text

from collections.abc import Iterable, Generator, Callable
from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path
from pydantic import AnyUrl as Url

from langchain.text_splitter import RecursiveCharacterTextSplitter


# XXX Must use Python 3.10!!!

# XXX Possibly replace Optional with | None

Markdown = str
Html = str
Embeddings = list[float]
Metadata = dict[str, Any]
Text = str

@dataclass
class TextChunk:
    id: str
    metadata: Metadata
    text: str

    # TODO Could remove this, I don't really need it since I decided not to serialize these.
    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "metadata": self.metadata,
            "text": self.text
        })

    @classmethod
    def from_json(cls, data: str | dict[str, Any]) -> "TextChunk":
        try:
            req_keys = ["id", "metadata", "text"]
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                raise ValueError("input must be a dict or json string")
            missing_keys = [rk for rk in req_keys if rk not in data]
            if missing_keys:
                raise ValueError(f"input data missing keys({missing_keys})")

            id = data.get("id")
            metadata = data.get("metadata")
            text = data.get("text")
            if not isinstance(id, str):
                raise ValueError("'id' must be str")
            if not isinstance(metadata, dict):
                raise ValueError("'metadata' must be type Metadata(dict[str, Any])")
            if not isinstance(text, Text):
                raise ValueError("'text' must be type str")

            return cls(id=id, metadata=metadata, text=text)

        except Exception as e:
            raise ValueError(f"failed to deserialize TextChunk: {e}")

ChunkBatch = list[TextChunk]

@dataclass
class VectorDbInput:
    ids: list[str]
    metadatas: list[Metadata]
    embeddings: list[Embeddings]
    texts: list[Text]

    @classmethod
    def from_chunk_batch_embeddings(cls, chunk_batch: ChunkBatch, embeddings: Embeddings) -> "VectorDbInput":
        try:
            if not isinstance(embeddings, list):
                raise ValueError("'embeddings' must be type Embeddings(list[float])")
            elen, cblen = len(embeddings), len(chunk_batch)
            if elen != cblen:
                raise ValueError(f"Embeddings len({elen}) must match ChunkBatch len({chlen})")
            return cls(
                ids=[chunk.id for chunk in chunk_batch],
                metadatas=[chunk.metadata for chunk in chunk_batch],
                texts=[chunk.text for chunk in chunk_batch],
                embeddings=embeddings
            )
        except Exception as e:
            raise ValueError(f"failed to produce VectorDbInput: {e}")

    @classmethod
    def from_json(cls, data: str | dict[str, Any]) -> "VectorDbInput":
        try:
            if isinstance(data, str):
                data = json.loads(data)
            return cls(
                ids=data["ids"],
                metadatas=data["metadatas"],
                embeddings=data["embeddings"],
                texts=data["texts"]
            )
            # TODO bruv, check errors.
        except Exception as e:
            raise ValueError(f"failed to deserialize VectorDbInput")

    def to_json(self) -> str:
        return json.dumps({
            "ids": self.ids,
            "metadatas": self.metadatas,
            "embeddings": self.embeddings,
            "texts": self.texts
        })


######## :bruv:
# TODO think about hashing the contents to detect changes in the source documents
# .... in a production situation, we would need to check for source differences

# TODO In production, we would need a way to detect document changes and update the vectordb
# TODO Think about the pattern were we only want cache hits from each *Processor


# XXX !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO XXX TODO XXX I think some of the process() or get() methods can raise exceptions
# .... that don't get caught in the failures list. Check on this.
# XXX 


class LameCache:
    def __init__(self, cache_path: Path, file_ext: str = "data"):
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path = cache_path
        self.ext = file_ext
        self.enc = "utf-8"  # XXX Only supports utf-8 encoded documents

    def _path(self, key: Any) -> Path:
        filename = hashlib.md5(str(key).encode("utf-8")).hexdigest()
        return self.cache_path / f"{filename}.{self.ext}"

    # TODO add "open()" method if needed

    def get(self, key: Any) -> Optional[str]:
        try:
            return self._path(key).read_text(encoding=self.enc)
        except FileNotFoundError:
            return None
        # TODO handle other errors
        return None

    def put(self, key: Any, data: str) -> None:
        # TODO check if we are actually writing utf-8 encoded docs
        with open(self._path(key), "w") as f:
            f.write(data)

    # TODO impl "remove_all_type" - find files *.ext and remove.
    # .... this handles the case of suffixs added.

    def remove(self, key: Any) -> None:
        os.remove(self._path(key))

    def exists(self, key: Any) -> bool:
        return os.path.exists(self._path(key))
        
    def reset(self):
        # TODO impl me?
        pass


class ChunkBatchProcessor:
    # TODO add option to force reprocessing, i.e., ignore cache
    def __init__(
        self,
        cache_path: Path | None,
        embed_func: Callable[[list[str]], list[list[float]]]
    ):
        self.cache = LameCache(cache_path, file_ext="vectordb_input.json") if cache_path else None
        self.embed_func = embed_func
        self.failures: list[dict[str, str]] = []


    def get_failures(self) -> list[dict[str, str]]:
        return self.failures

    def process(
        self, chunk_batches: Iterable[tuple[str, ChunkBatch]]
    ) -> Generator[tuple[str, VectorDbInput], None, None]:
        self.failures = []
        for key, chunk_batch in chunk_batches:
            try:
                if self.cache:
                    cache_res = self.cache.get(key)
                    if cache_res:
                        yield key, VectorDbInput.from_json(cache_res)
                        continue
                
                embeddings = self.embed_func([text_chunk.text for text_chunk in chunk_batch])
                vectordb_input = VectorDbInput.from_chunk_batch_embeddings(chunk_batch, embeddings)
                if self.cache:
                    self.cache.put(key, vectordb_input.to_json())
                yield key, vectordb_input
            except Exception as e:
                self.failures.append({key: f"error processing embeddings for ChunkBatch {e}"})


class TextProcessor:
    # TODO impl metadata processor
    # .... Match for: https://github.com/chroma-core/chroma/pull/2918
    def __init__(
        self,
        batch_size: int = 32,
        chunk_size: int = 512,
        chunk_overlap: int = 100
    ):
        self.failures: list[dict[str, str]] = []

        # TODO think about passing in a callable here.
        self.batch_size = batch_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def get_failures(self) -> list[dict[str, str]]:
        return self.failures

    def _process(self, key: str, text: Text) -> Generator[ChunkBatch, None, None]:
        batch: ChunkBatch = []

        for chunk_idx, chunk in enumerate(self.splitter.split_text(text)):
            doc_id = f"{key}-c{chunk_idx}"
            try:
                # TODO maybe generate more metadata? Like that PR?
                batch.append(TextChunk.from_json({
                    "id": doc_id,
                    "metadata": {"source": str(key), "part": chunk_idx},
                    "text": str(chunk)
                }))
            except Exception as e:
                raise ValueError(f"error processing chunk {chunk_idx}: {e}")

            if (len(batch) + 1) % (self.batch_size + 1) == 0:
                yield batch
                batch = []

        if len(batch):
            yield batch

    def process(
        self, keys_texts: Iterable[tuple[str, Text]]
    ) -> Generator[tuple[str, ChunkBatch], None, None]:
        self.failures = []
        for key, text in keys_texts:
            try:
                chunks: ChunkBatch = self._process(key, text)
                for chunk in chunks:
                    # XXX this is going to have the same key for multiple chunks.
                    yield key, chunk
            except Exception as e:
                self.failures.append({str(key): e})
                continue


class HtmlProcessor:
    # TODO Think about using BeautifySoup to preprocess HTML
    # TODO Thinking about the possibility of multiprocessing...
    # .... I think I need the queue to read from the cache... possibly.
    # .... maybe not ...
    def __init__(self, cache_path: Optional[Path]):
        self.cache = LameCache(cache_path, file_ext="markdown") if cache_path else None
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = False
        self.converter.ignore_images = False  # Should we though?
        self.converter.body_width = 0

        self.failures: list[dict[Url, str]] = []

    def process(self, url: Url | str, html: Html) -> Optional[Markdown]:
        if isinstance(url, str):
            url = Url(url)

        if self.cache:
            cache_res = self.cache.get(url)
            if cache_res is not None:
                return cache_res

        markdown: Optional[Markdown] = None
        try:
            markdown = self.converter.handle(html)
        except Exception as e:
            self.failures.append({str(url): e})

        if self.cache and markdown:
            self.cache.put(url, markdown)

        return markdown


    def process_all(
        self, urls_htmls: Iterable[tuple[Url | str, Html]]
    ) -> Generator[tuple[Url, Markdown], None, None]:
        self.failures = []
        for url, html in urls_htmls:
            markdown = self.process(url, html)
            if markdown is None:
                continue
            yield url, markdown

    def get_failures(self) -> list[dict[Url, str]]:
        return self.failures


class WebScraper:
    # TODO Currently assumes HTML responses. Support other content types.
    # TODO probably want timing counters at some point
    # TODO need a way to back off if getting temp fail codes on GETs
    # TODO flag for cache on/off, not needed now.
    # TODO need flag to enable check for cache coherency.
    # TODO should serialize failures to file in cache dir to retry later.
    def __init__(self, cache_path: Optional[Path]):
        self.cache = LameCache(cache_path, file_ext="html") if cache_path else None
        self.fetch_delay = 690
        self.last_fetch = None
        self.failures: list[dict[Url, str]] = []

    def _check_url(self, url: Url | str) -> Url:
        return url
        
    def _get(self, url: Url) -> Optional[Html]:
        # Try to not get blocked for fetching too quickly.
        if self.last_fetch:
            fetch_diff = time.perf_counter() - self.last_fetch
            if fetch_diff < self.fetch_delay:
                time.sleep(fetch_diff)
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            self.failures.append({str(url): e})
            return None
        finally:
            self.last_fetch = time.perf_counter()

    def get(self, url: Url | str) -> Optional[Html]:
        """
        Returns the HTML for a given URL or None on error.

        Call get_failures() to see failure.
        """
        if isinstance(url, str):
            url = Url(url)

        if self.cache:
            cache_res = self.cache.get(url)
            if cache_res is not None:
                return cache_res
        
        data = self._get(url)

        if self.cache and data:
            self.cache.put(url, data)

        return data

    def get_all(self, urls: Iterable[Url | str]) -> Generator[tuple[Url | str, Html], None, None]:
        """
        Gets all urls and yields data for each.

        Call get_failures() to see failures.
        """
        self.failures = []  # XXX perhaps should warn if non-empty
        for url in urls:
            data = self.get(url)
            if data is None:
                continue
            yield url, data

    def get_failures(self) -> list[dict[Url, str]]:
        return self.failures
    


