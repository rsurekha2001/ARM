
# Import necessary libraries
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import RAG utility functions
from rag_utils import retrieve, rag, DEFAULT_K


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model and the saved FAISS vector store
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True   # the index is our own file, built in the notebook
)

print(f"Vectors stored: {vectorstore.index.ntotal}")


# ---------------------------------------------------------
# Helper: read and validate the JSON body
# ---------------------------------------------------------

def read_params():
    """Return (params, error_response). error_response is None when the request is valid."""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return None, (jsonify({"error": "Field 'query' is required."}), 400)
    try:
        k = max(1, min(int(data.get("k", DEFAULT_K)), 20))
    except (TypeError, ValueError):
        return None, (jsonify({"error": "Field 'k' must be an integer."}), 400)
    return {
        "query": query,
        "k": k,
        "model_name": data.get("model_name", "gpt-4o-mini"),
        "temperature": data.get("temperature", 0.0),
        "top_p": data.get("top_p", 1.0),
        "max_tokens": data.get("max_tokens", 512),
    }, None


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """Example request: {"query": "What is the cancellation policy?", "k": 5}"""
    params, error = read_params()
    if error:
        return error

    chunks = retrieve(query=params["query"], k=params["k"], vectorstore=vectorstore)
    return jsonify({"query": params["query"], "k": params["k"], "relevant_chunks": chunks})


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """Returns the generated answer together with the chunks used to produce it."""
    params, error = read_params()
    if error:
        return error

    answer, chunks = rag(vectorstore=vectorstore, **params)
    return jsonify({**params, "relevant_chunks": chunks, "answer": answer})


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """Returns only the generated answer."""
    params, error = read_params()
    if error:
        return error

    generated_answer, _ = rag(vectorstore=vectorstore, **params)
    return jsonify({"answer": generated_answer})


# ---------------------------------------------------------
# Run the Flask application (local use only; the container uses gunicorn)
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(debug=False)
