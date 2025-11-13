import uuid
def create_ticket(title, severity, summary):
    return {
        "ticket_id": f"T-{uuid.uuid4().hex[:6].upper()}" ,
        "title": title,
        "severity": severity,
        "summary": summary
    }