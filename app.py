import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from utils.ticket_tool import create_ticket
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SupportGenie AI Assistant",
    page_icon="🧞",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load knowledge base (cached)
@st.cache_data
def load_kb():
    kb_path = Path("data/kb_seed/support_kb.json")
    with kb_path.open("r", encoding="utf-8") as f:
        return json.load(f)

kb_data = load_kb()

# Initialize embedder and FAISS index (cached)
@st.cache_resource
def initialize_embeddings(kb_data):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    entries = [entry['content'] for entry in kb_data]
    embeddings = embedder.encode(entries, convert_to_numpy=True)
    
    dimensions = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(embeddings)
    
    return embedder, index

embedder, index = initialize_embeddings(kb_data)

# Retrieval function 
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

# Initialize OpenAI client
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# QA function
def answer_query(query, k=3):
    results = retrieve(query, k)

    context = "\n".join([f"[{d['id']}] {d['content']}" for d in results])

    system_prompt = (
        "You are SupportGenie, an AI support assistant. "
        "- Retrieve answers from the knowledge base and cite document IDs. "
        "- Be concise, professional, and avoid hallucinations. "
        "- If unsure, say: 'That information isn't available in the knowledge base.' "
    )

    user_prompt = f"Knowledge Base:\n{context}\n\nUser: {query}\nAnswer:"

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return answer, results

# Ticket generation function
def generate_ticket_title(query):
    systemPrompt = ("You are a support ticket assistant. Generate a clear,"
                    "concise ticket title (max 60 characters) from the user's"
                    "issue description. Be specific and actionable."
                    "- Don't include 'severity' in the title."
                    )
    
    userPrompt = f"User: \n{query}\nTitle:"
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ],
            temperature=0.3,
        )
        title = response.choices[0].message.content.strip()
        return title
    
    except Exception:
        return "User Generated Ticket"

# Handle user input
def handle_user_input(query):
    q_lower = query.lower()
    if "open" in q_lower and "ticket" in q_lower or "report" in q_lower and "issue" in q_lower:
        if "high" in q_lower:
            severity = "high"
        elif "low" in q_lower:
            severity = "low"
        elif "medium" in q_lower:
            severity = "medium"
        else:
            severity = "unspecified"
            
        title = generate_ticket_title(query)
        summary = query
        
        ticket = create_ticket(title, severity, summary)
        return f"**Ticket Created**\n\n**ID:** {ticket['ticket_id']}\n\n**Title:** {ticket['title']}\n\n**Severity:** {ticket['severity']}", None
    
    return answer_query(query)

# ============ STREAMLIT UI ============

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### SupportGenie")
    st.markdown("**Agentic RAG Prototype**")
    st.markdown("Ask questions or report issues!")
    
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


st.markdown('<div class="main-header">SupportGenie AI Assistant</div>', unsafe_allow_html=True)
st.markdown("---")

# Check API key
if not openai_api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
    st.stop()
else:
    st.success(f"System initialized | Knowledge Base entries: {len(kb_data)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Knowledge Base Sources"):
                for src in message["sources"]:
                    st.markdown(f"**[{src['id']}]** {src['title']} _(distance: {src['distance']:.3f})_")

# Chat input
if prompt := st.chat_input("Ask a question or report an issue..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, sources = handle_user_input(prompt)
                st.markdown(response)
                
                # Show sources
                if sources:
                    with st.expander("Knowledge Base Sources"):
                        for src in sources:
                            st.markdown(f"**[{src['id']}]** {src['title']} _(distance: {src['distance']:.3f})_")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": None
                })
