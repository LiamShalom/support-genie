import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from utils.ticket_tool import create_ticket

kb_path = Path("data/kb_seed/support_kb.json")

with kb_path.open("r", encoding="utf-8") as f:
    kb_data = json.load(f)

# Print entries for verification
# print(f"Loaded {len(kb_data)} knowledge base entries.")
# for entry in kb_data:  
#     print(entry)

# small, very fast, good for short FAQs
embedder = SentenceTransformer('all-MiniLM-L6-v2')

entries = [entry['content'] for entry in kb_data]

embeddings = embedder.encode(entries, convert_to_numpy=True)

dimensions = embeddings.shape[1]
index = faiss.IndexFlatL2(dimensions)
index.add(embeddings)

# Example query for testing
# query = "How do I reset my password?"
# query_emb = embedder.encode([query], convert_to_numpy=True)
# D, I = index.search(query_emb, k=3)
# print("Top matches:", [kb_data[i]["id"] for i in I[0]])

def retrieve(query, k=3):
    query_embed = embedder.encode([query], convert_to_numpy=True)

    D, I = index.search(query_embed, k)

    results = []
    for rank, i in enumerate(I[0]):
        entry = kb_data[i]
        results.append({
            "rank": rank + 1,
            "id": entry["id"],
            "title": entry["title"],
            "content": entry["content"],
            "distance": float(D[0][rank])
        })
    return results

# Example usage of retrieve function:
# query = "How do I reset my password if I lost my phone?"
# results = retrieve(query)

# for r in results:
#     print(f"[{r['id']}] ({r['distance']:.2f}) - {r['title']}")

# Requires OpenAI API key in environment or .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def answer_query(query, k=3):
    results = retrieve(query, k)

    context = "\n".join([f"[{d['id']}] {d['content']}" for d in results])

    system_prompt = (
        "You are SupportGenie, an AI support assistant. "
        "- Retrieve answers from the knowledge base and cite document IDs. "
        "- If the user says 'open ticket' or 'report issue' , call the tool `create_ticket` with title, severity, and summary. "
        "- Be concise, professional, and avoid hallucinations. "
        "- If unsure, say: 'That information isn't available in the knowledge base.' "
    )

    user_prompt = f"Knowledge Base:\n{context}\n\nUser: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return answer

def handle_user_input(query):
    q_lower = query.lower()
    if "open ticket" in q_lower or "report issue" in q_lower:
        if "high" in q_lower:
            severity = "high"
        elif "low" in q_lower:
            severity = "low"
        else:
            severity = "medium"  # default severity
            
        title = "User Reported Issue"
        summary = query
        
        ticket = create_ticket(title, severity, summary)
        return f"Ticket created with ID: {ticket['ticket_id']}, Title: {ticket['title']}, Severity: {ticket['severity']}"
    
    return answer_query(query)

print(answer_query("How do I reset my password if I lost my phone?"))

print(handle_user_input("Open a ticket for this SSO issue with severity high"))