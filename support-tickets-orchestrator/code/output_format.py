import csv
import os  # FIXED: was missing — caused NameError crash on line 3

OUTPUT_CSV = os.path.join("..", "support_tickets", "output.csv")
VALID_STATUS = {"replied", "escalated"}
VALID_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

errors = []
with open(OUTPUT_CSV, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if row.get("status") not in VALID_STATUS:
            errors.append(f"Row {i+1}: invalid status '{row.get('status')}'")
        if row.get("request_type") not in VALID_TYPES:
            errors.append(f"Row {i+1}: invalid request_type '{row.get('request_type')}'")
        if not row.get("response", "").strip():
            errors.append(f"Row {i+1}: empty response")
        if not row.get("product_area", "").strip():
            errors.append(f"Row {i+1}: empty product_area")
        if not row.get("justification", "").strip():
            errors.append(f"Row {i+1}: empty justification")

if errors:
    print(f"❌ {len(errors)} format errors found:")
    for e in errors:
        print(f"  {e}")
else:
    print(f"✅ Output CSV format is valid — all {i+1} rows pass format checks")