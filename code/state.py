from typing import TypedDict, Optional

class TicketState(TypedDict):
    """
    Shared state for the agentic pipeline. 
    Stores both the raw CSV input and the derived AI decisions.
    """
    # Inputs from CSV
    issue: str
    subject: str
    company: str

    # Intermediate AI Data
    search_query: str      # Technical keywords for database lookup
    retrieved_context: str # The actual text found in the Markdown docs
    is_context_relevant: str # "yes" or "no" (CRAG logic)
    
    # Required Hackathon Outputs
    request_type: str     # product_issue, feature_request, bug, invalid
    product_area: str     # The specific sub-domain (e.g., 'billing', 'lost_card')
    status: str           # 'replied' or 'escalated'
    response: str         # The final grounded answer
    justification: str    # Technical reasoning for the decision