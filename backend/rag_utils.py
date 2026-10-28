
import os
from langchain_openai import ChatOpenAI
from prompts import build_prompt          # the same prompt that was evaluated in the notebook

DEFAULT_K = 8                              # keep in sync with K in the notebook


# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = DEFAULT_K, vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]


# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:
    return model.invoke(build_prompt(query, retrieved_chunks))


# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = DEFAULT_K,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)

    return answer.content, retrieved_chunks
