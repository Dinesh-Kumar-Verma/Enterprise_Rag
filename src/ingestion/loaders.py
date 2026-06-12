"""
Multi-source document ingestion pipeline.
Supports: PDF, DOCX, TXT, CSV, web URLs, REST APIs, plain text.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from loguru import logger

# from config.settings import get_settings

# settings = get_settings()


def _doc_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


class DocumentLoader:
    """Unified loader for all supported source types."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

    def load_file(self, path: str | Path) -> list[Document]:
        path = Path(path)
        ext = path.suffix.lower()
        logger.info(f"Loading file: {path.name} ({ext})")

        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(path))
        elif ext in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(str(path))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = loader.load()
        for doc in docs:
            doc.metadata.update(
                {
                    "source_type": "file",
                    "source_name": path.name,
                    "file_path": str(path),
                    "doc_hash": _doc_hash(doc.page_content),
                }
            )
        return docs

    def load_url(self, url: str) -> list[Document]:
        logger.info(f"Loading URL: {url}")
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "EnterpriseRAG/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)

            domain = urlparse(url).netloc
            doc = Document(
                page_content=text,
                metadata={
                    "source_type": "web",
                    "source_name": domain,
                    "url": url,
                    "doc_hash": _doc_hash(text),
                },
            )
            return [doc]
        except Exception as e:
            logger.error(f"Failed to load URL {url}: {e}")
            return []

    def load_api(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        text_field: str = "content",
        params: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Load documents from a REST API endpoint (JSON response)."""
        logger.info(f"Loading API: {url}")
        try:
            resp = requests.get(url, headers=headers or {}, params=params or {}, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            items = data if isinstance(data, list) else data.get("results", data.get("items", [data]))

            docs = []
            for item in items:
                if isinstance(item, dict):
                    text = item.get(text_field, str(item))
                else:
                    text = str(item)

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_type": "api",
                            "source_name": urlparse(url).netloc,
                            "api_url": url,
                            "doc_hash": _doc_hash(text),
                        },
                    )
                )
            return docs
        except Exception as e:
            logger.error(f"Failed to load API {url}: {e}")
            return []

    def load_text(self, text: str, source_name: str = "manual") -> list[Document]:
        doc = Document(
            page_content=text,
            metadata={
                "source_type": "text",
                "source_name": source_name,
                "doc_hash": _doc_hash(text),
            },
        )
        return [doc]

    def load_directory(self, directory: str | Path) -> list[Document]:
        directory = Path(directory)
        docs = []
        for path in directory.rglob("*"):
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS and path.is_file():
                try:
                    docs.extend(self.load_file(path))
                except Exception as e:
                    logger.warning(f"Skipping {path.name}: {e}")
        logger.info(f"Loaded {len(docs)} documents from {directory}")
        return docs

def main():
    loader = DocumentLoader()
    # Example usage:
    # docs_from_file = loader.load_file(r"C:\Users\Dinesh Verma\Downloads\Two Pointer Pattern.pdf")
    # docs_from_url = loader.load_url("https://example.com")
    # docs_from_api = loader.load_api("https://api.example.com/data", text_field="description")
    # docs_from_text = loader.load_text("This is some sample text.", source_name="sample_input")
    # docs_from_directory = loader.load_directory("path/to/documents/")
    
if __name__ == "__main__":
    main()