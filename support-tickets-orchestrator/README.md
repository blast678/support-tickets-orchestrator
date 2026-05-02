
---

# 🤖 Support Tickets Orchestrator

A state-of-the-art, terminal-based AI agent designed to triage, route, and answer complex support tickets across three distinct product ecosystems: **HackerRank**, **Claude**, and **Visa**. This system utilizes an agentic state-machine architecture to ensure 100% groundedness and safety.

## 🏗️ Architectural Overview: The Agentic Assembly Line

Unlike standard chatbots, this system is built as a **Deterministic State Machine** orchestrated via **LangGraph**. Every ticket passes through a specialized pipeline where safety and relevance are audited at every step.

* **Adaptive Router**: Translates messy human input into technical search queries and identifies domain intent.
* **Contextual Retriever**: Performs high-precision search across localized **FAISS** vector databases using **Maximal Marginal Relevance (MMR)** to ensure information diversity.
* **CRAG Grader**: A "Corrective RAG" auditor that evaluates document relevance and forces human escalation for high-risk or unsupported cases.
* **CoVe Generator**: A "Chain-of-Verification" node that generates responses only after writing a cited justification to prevent hallucination.

## 🔬 Core Research Techniques

| Technique                    | Implementation                    | Impact                                                                                               |
| :--------------------------- | :-------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Contextual Retrieval**     | Metadata Injection in `ingest.py` | Prepends document hierarchies to chunks to prevent domain confusion (e.g., Visa vs. Claude billing). |
| **Diverse Search (MMR)**     | Maximal Marginal Relevance        | Ensures the LLM receives a diverse set of facts for complex, multi-part questions.                   |
| **CRAG**                     | Self-Reflective Auditing          | Acts as a firewall; if retrieved info is insufficient, the agent escalates instead of guessing.      |
| **CoVe**                     | Reasoning-First Generation        | Forces the model to cite specific source files and passages *before* drafting a response.            |
| **Deterministic Guardrails** | Hybrid Regex-LLM Filtering        | Bypasses AI entirely for 100% reliability in cases of fraud, theft, or identity crimes.              |

## 📁 Repository Structure

```text
.
├── code/
│   ├── ingest.py         # Knowledge base construction (Contextual Ingestion)
│   ├── main.py           # LangGraph orchestrator and entry point
│   ├── nodes.py          # Logic for Router, Grader, and Generator agents
│   ├── retriever.py      # MMR-based search interface
│   ├── state.py          # Global TicketState schema
│   └── output_format.py  # Integrity validation script
├── support_tickets/
│   ├── support_tickets.csv # Input tickets
│   └── output.csv          # Agent predictions (Final Output)
├── data/                 # Local support corpus (Markdown)
├── faiss_index/          # Local vector databases
└── requirements.txt      # Project dependencies
```

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10+
* A Groq API Key (Llama 3.3-70B recommended for high-quality reasoning)

### 2. Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/blast678/support-tickets-orchestrator.git
   cd support-tickets-orchestrator
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   Create a `.env` file in the root directory and add your key:

   ```text
   GROQ_API_KEY=your_key_here
   ```

### 3. Execution

1. **Ingest the Data**: Build the contextual indices (Run once):

   ```bash
   python code/ingest.py
   ```

2. **Run the Agent**: Process the support tickets:

   ```bash
   python code/main.py
   ```

3. **Validate Output**: Ensure the final CSV format is correct:

   ```bash
   python code/output_format.py
   ```

## 📊 Evaluation & Safety

The agent is optimized to maximize scores across the **HackerRank Orchestrate** dimensions:

* **Groundedness**: Verified through CoVe reasoning and direct source citations in the `justification` column.
* **Safety**: High-risk financial and security keywords trigger immediate human escalation via deterministic Regex guardrails.
* **Engineering Hygiene**: Deterministic outputs are ensured by fixed random seeds and zero-temperature configurations.

---
