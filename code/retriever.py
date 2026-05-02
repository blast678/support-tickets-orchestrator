import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from state import TicketState

# Load embeddings once for the entire session
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_cache = {}

def get_faiss_index(company: str):
    """Retrieves the local index, using caching for speed."""
    comp = company.lower()
    if comp in vector_cache: return vector_cache[comp]
    
    path = os.path.join("faiss_index", comp)
    if os.path.exists(path):
        index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        vector_cache[comp] = index
        return index
    return None

def retrieve_context(state: TicketState) -> dict:
    print(f"--- [NODE] RETRIEVER ({state['company']}) ---")
    
    query = state.get("search_query", state["issue"])
    
    # Try primary company index first
    index = get_faiss_index(state.get("company", "hackerrank"))
    if index:
        docs = index.max_marginal_relevance_search(query, k=6, fetch_k=15)
        if docs:
            context_str = "\n\n".join([f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in docs])
            return {"retrieved_context": context_str}
    
    # Fallback: search ALL indexes and merge top results
    print(f"--- [NODE] RETRIEVER: primary index miss, trying all indexes ---")
    all_docs = []
    for company in ["hackerrank", "claude", "visa"]:
        fallback_index = get_faiss_index(company)
        if fallback_index:
            try:
                docs = fallback_index.max_marginal_relevance_search(query, k=2, fetch_k=6)
                all_docs.extend(docs)
            except: continue
    
    if all_docs:
        context_str = "\n\n".join([f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in all_docs])
        return {"retrieved_context": context_str}
    
    return {"retrieved_context": "ERROR: No relevant documentation found in any index."}