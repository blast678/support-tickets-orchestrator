import csv
import os
from langgraph.graph import StateGraph, END
from state import TicketState
from nodes import triage_router, document_grader, generator_node
from retriever import retrieve_context

INPUT_CSV = os.path.join("..", "support_tickets", "support_tickets.csv") 
OUTPUT_CSV = os.path.join("..", "support_tickets", "output.csv")

def build_graph():
    workflow = StateGraph(TicketState)
    workflow.add_node("router", triage_router)
    workflow.add_node("retriever", retrieve_context)
    workflow.add_node("grader", document_grader)
    workflow.add_node("generator", generator_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "grader")
    workflow.add_conditional_edges("grader", lambda x: "generate", {"generate": "generator"})
    workflow.add_edge("generator", END)
    return workflow.compile()

def process_tickets():
    app = build_graph()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    with open(INPUT_CSV, mode='r', encoding='utf-8-sig') as infile, \
         open(OUTPUT_CSV, mode='w', encoding='utf-8-sig', newline='') as outfile:
        
        # FIXED: Clean keys to handle trailing spaces or case differences
        reader = csv.DictReader(infile)
        reader.fieldnames = [k.strip().lower() for k in reader.fieldnames]
        
        writer = csv.DictWriter(outfile, fieldnames=["status", "product_area", "response", "justification", "request_type"])
        writer.writeheader()
        
        for i, row in enumerate(reader):
            print(f"\n--- Processing Ticket {i+1} ---")
            
            # FIXED: Robust key access
            issue = row.get("issue", "")
            subject = row.get("subject", "")
            company = row.get("company", "None") or "None"
            
            # Run the agentic pipeline
            final_state = app.invoke({"issue": issue, "subject": subject, "company": company})
            
            writer.writerow({
                "status": final_state.get("status"),
                "product_area": final_state.get("product_area"),
                "response": final_state.get("response"),
                "justification": final_state.get("justification"),
                "request_type": final_state.get("request_type")
            })
            print(f"✅ Success: Saved result for Ticket {i+1}")

if __name__ == "__main__":
    process_tickets()