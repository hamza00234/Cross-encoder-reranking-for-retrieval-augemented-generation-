from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Splits source documents into overlapping fixed-size chunks using
    LangChain's RecursiveCharacterTextSplitter with sentence-boundary awareness.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        chunks: list[dict] = []
        for doc in documents:
            parts = self.splitter.split_text(doc["content"])
            for chunk_index, content in enumerate(parts):
                chunks.append(
                    {
                        "chunk_id": f"{doc['document_id']}_chunk_{chunk_index}",
                        "document_id": doc["document_id"],
                        "source_title": doc["source_title"],
                        "chunk_index": chunk_index,
                        "content": content,
                    }
                )
        return chunks
