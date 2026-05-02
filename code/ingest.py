import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_database():
    """Builds a local, context-enriched FAISS index for each company."""
    print("--- 🚀 STARTING CONTEXTUAL INGESTION ---")
    
    # 1. Initialize local embeddings (Q1 2025 Industry Standard for local RAG)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Setup Markdown splitting logic
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for company in ["hackerrank", "claude", "visa"]:
        data_dir = os.path.join("..", "data", company)
        if not os.path.exists(data_dir):
            print(f"Skipping {company}: Directory not found.")
            continue
            
        all_splits = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.md'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # CONTEXTUAL ENRICHMENT: Prepend hierarchy to text
                        md_header_splits = markdown_splitter.split_text(content)
                        splits = text_splitter.split_documents(md_header_splits)
                        for split in splits:
                            # Capture header path (e.g., 'Lost Card > Reporting')
                            headers = " > ".join([v for k, v in split.metadata.items() if k.startswith("H")])
                            # Physical context injection into the string before embedding
                            split.page_content = f"Company: {company} | Area: {headers}\n{split.page_content}"
                            split.metadata["source"] = file
                        all_splits.extend(splits)
                    except: continue
        
        if all_splits:
            # 3. Save FAISS index locally
            vectorstore = FAISS.from_documents(all_splits, embeddings)
            os.makedirs("faiss_index", exist_ok=True)
            vectorstore.save_local(os.path.join("faiss_index", company))
            print(f"✅ Success: Contextual Index built for {company}")

if __name__ == "__main__":
    build_vector_database()