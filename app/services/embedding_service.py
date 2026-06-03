from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, runs locally, ~60-80ms per query
EMBEDDING_DIM = 384

# BGE models are trained asymmetrically: queries should be prefixed, documents should not.
# See https://huggingface.co/BAAI/bge-small-en-v1.5
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"⏳ [EMBEDDING] Loading local model '{EMBEDDING_MODEL}'...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ [EMBEDDING] Model loaded")
    return _model


def generate_embedding(text: str) -> list[float]:
    text = text.strip().replace("\n", " ")
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    cleaned = [t.strip().replace("\n", " ") for t in texts]
    model = get_model()
    embeddings = model.encode(cleaned, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


def generate_query_embedding(text: str) -> list[float]:
    cleaned = text.strip().replace("\n", " ")
    model = get_model()
    embedding = model.encode(BGE_QUERY_PREFIX + cleaned, normalize_embeddings=True)
    return embedding.tolist()


def generate_query_embeddings_batch(texts: list[str]) -> list[list[float]]:
    cleaned = [BGE_QUERY_PREFIX + t.strip().replace("\n", " ") for t in texts]
    model = get_model()
    embeddings = model.encode(cleaned, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]
