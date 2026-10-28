
RULES = """
1. Use only information explicitly supported by the retrieved policy information.
2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
3. Include all relevant information needed to answer the query completely.
4. Do not omit important conditions, exceptions, limitations, deadlines, or fees.
5. Quote numbers, times, and fees exactly as written. Never estimate or round them.
6. If the question names a specific property, rate plan, or room type, use only the information that applies to it.
7. Do not include unrelated information from the retrieved policy information.
8. Answer the user's specific question directly and ensure the response addresses what was asked.
9. Keep the answer concise, clear, and easy for hotel staff to relay to a guest.
10. Do not invent policies, fees, limits, or conditions.
11. If the information is insufficient, say:
    "The available policy documents do not contain enough information to answer this question."
    If the retrieved information conflicts, clearly mention the conflict.
12. Do not add advice, caveats, or hedging words (for example "typically", "generally", "usually")
    unless they are stated in the retrieved policy information.
"""


def build_prompt(query, retrieved_chunks):
    """Build the generation prompt. retrieved_chunks is a list of chunk strings."""
    if isinstance(retrieved_chunks, (list, tuple)):
        context = "\n\n---\n\n".join(retrieved_chunks)
    else:
        context = str(retrieved_chunks)

    return f"""You are an AI assistant answering questions about AuroraStay hotel policies for hotel staff.

User Query:
{query}

Retrieved Policy Information:
{context}

Rules:
{RULES}

Answer:
"""
