from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from textwrap import shorten

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings


COLLECTION_NAME = "case_2_rag_assistant"
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


class LocalHashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Small local embedding function for learning RAG without API keys."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower())

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше 0")
    if overlap < 0:
        raise ValueError("overlap не может быть отрицательным")
    if overlap >= chunk_size:
        raise ValueError("overlap должен быть меньше chunk_size")

    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def recreate_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    embedding_function = LocalHashEmbeddingFunction()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(
    client: chromadb.PersistentClient,
    text_path: Path,
    chunk_size: int,
    overlap: int,
) -> tuple[chromadb.Collection, list[str]]:
    text = load_text(text_path)
    chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        raise ValueError(f"Файл {text_path} пустой или не содержит текста")

    collection = recreate_collection(client)
    collection.add(
        ids=[f"chunk-{index}" for index in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": str(text_path), "chunk": index}
            for index in range(len(chunks))
        ],
    )

    return collection, chunks


def extractive_answer(best_chunk: str, question: str) -> str:
    if not best_chunk:
        return "Не удалось найти подходящий фрагмент в базе знаний."

    sentences = re.split(r"(?<=[.!?])\s+", best_chunk.strip())
    question_tokens = set(TOKEN_RE.findall(question.lower()))

    best_index = 0
    best_score = -1
    for index, sentence in enumerate(sentences):
        sentence_tokens = TOKEN_RE.findall(sentence.lower())
        score = sum(1 for token in sentence_tokens if token in question_tokens)
        if score > best_score:
            best_index = index
            best_score = score

    answer_sentences = sentences[best_index : best_index + 2]
    answer = " ".join(answer_sentences).strip()
    return answer or best_chunk


def run_rag(args: argparse.Namespace) -> None:
    client = chromadb.PersistentClient(
        path=str(args.db_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    collection, chunks = build_index(
        client=client,
        text_path=args.text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    results = collection.query(query_texts=[args.question], n_results=args.top_k)
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    print("=== Case 2: RAG assistant with ChromaDB ===")
    print(f"Текстовая база: {args.text}")
    print(f"Папка ChromaDB: {args.db_dir}")
    print(f"Создано чанков: {len(chunks)}")
    print(f"Вопрос: {args.question}")
    print()

    print("Ответ ассистента на основе лучшего найденного чанка:")
    print(extractive_answer(documents[0], args.question))
    print()

    print("Найденные фрагменты:")
    for rank, (document, distance, metadata) in enumerate(
        zip(documents, distances, metadatas),
        start=1,
    ):
        chunk_number = metadata["chunk"]
        preview = shorten(document.replace("\n", " "), width=420, placeholder="...")
        print(f"{rank}. chunk-{chunk_number} | cosine distance: {distance:.4f}")
        print(f"   {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Учебный RAG-ассистент: текст -> чанки -> эмбеддинги -> поиск.",
    )
    parser.add_argument(
        "--text",
        type=Path,
        default=Path("data/knowledge_base.txt"),
        help="Путь к текстовой базе знаний.",
    )
    parser.add_argument(
        "--question",
        "-q",
        default="Что такое чанкинг и зачем он нужен в RAG?",
        help="Вопрос, по которому нужно найти релевантные чанки.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=90,
        help="Размер чанка в словах.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=20,
        help="Перекрытие соседних чанков в словах.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Сколько похожих чанков показать.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("chroma_db"),
        help="Папка для локальной базы ChromaDB.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_rag(parse_args())
