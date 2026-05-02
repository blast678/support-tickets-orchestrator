# Agentic Support Triage System

## Architecture
This agent uses a **Corrective RAG (CRAG)** framework orchestrated via **LangGraph**.
1. **Router**: Classifies intent and generates optimized search queries.
2. **Retriever**: Uses **Contextual Retrieval** and FAISS to pull diverse chunks from the corpus.
3. **Grader**: A strict auditor that escalates high-risk or unsupported tickets.
4. **Generator**: Uses **Chain-of-Verification** to ensure responses are grounded in retrieved docs.

## Setup
1. Place `GROQ_API_KEY` in the root `.env` file.
2. Install dependencies: `pip install -r ../requirements.txt`
3. Shorten the long Claude filename in `data/` to avoid Windows path errors.

## Execution
1. **Ingest**: `python ingest.py` (Run once to build local FAISS indices).
2. **Process**: `python main.py` (Processes `support_tickets.csv` and outputs to `output.csv`).