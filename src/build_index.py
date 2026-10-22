# build the FAISS index over the guideline corpus (protocol 2.4)
# chunks embedded with config.EMBEDDING_MODEL, L2-normalised, exact inner-product index
# -> search score = cosine similarity
# id_map.json = chunk_id for each index position
# run again after every corpus rebuild (old index silently gives wrong chunks)
#
# usage:
#   python src/build_index.py

import json

import faiss
import numpy as np

from config import CORPUS_FILE, EMBEDDING_MODEL, FAISS_INDEX, ID_MAP_FILE, INDEX_DIR


def load_corpus():
    chunk_ids = []
    texts = []
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                chunk_ids.append(r["chunk_id"])
                texts.append(r["text"])
    return chunk_ids, texts


def embed(texts, model=None):
    # normalised float32 vectors, used for chunks here and for queries in run_rag.py
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(list(texts), batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
    return vectors.astype(np.float32)


def main():
    chunk_ids, texts = load_corpus()
    print(f"{len(texts)} chunks from {CORPUS_FILE.name}; embedding with {EMBEDDING_MODEL}")
    vectors = embed(texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX))
    ID_MAP_FILE.write_text(json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {FAISS_INDEX.name} ({index.ntotal} vectors, dim {index.d}) and {ID_MAP_FILE.name}")


if __name__ == "__main__":
    main()
