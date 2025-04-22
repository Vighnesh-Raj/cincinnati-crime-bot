# === STREAMLIT VERSION: CINCINNATI CRIME CHATBOT WITH FLAN-T5-LARGE ===
import pandas as pd
import time
import mlflow
import re
from collections import Counter
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download, login
import os

# === Hugging Face Auth ===
login(token=os.environ.get("HF_TOKEN"))

# === Setup ===
st.set_page_config(page_title="Cincinnati Crime Chatbot", page_icon="🚓")
st.title("🚔 Cincinnati Crime Chatbot")
st.markdown("Ask about recent police activity in your neighborhood.")

# === Load FLAN-T5-Large Model ===
st.write("Loading FLAN-T5-Large...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# === Load & Clean Dataset ===
@st.cache_data(show_spinner="Fetching latest police reports...")
def load_data():
    path = hf_hub_download(
        repo_id="mlsystemsg1/cincinnati-crime-data",
        repo_type="dataset",
        filename="calls_for_service_latest.csv"
    )
    df = pd.read_csv(path, low_memory=False)
    df['incident_type_desc'] = df['incident_type_desc'].fillna(df['incident_type_id'])
    df['priority'] = pd.to_numeric(df['priority'], errors='coerce')
    df['create_time_incident'] = pd.to_datetime(df['create_time_incident'], errors='coerce')
    for col in ['sna_neighborhood', 'cpd_neighborhood']:
        df[col] = df[col].astype(str).str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)
    df['neighborhood'] = df['cpd_neighborhood'].combine_first(df['sna_neighborhood'])
    df = df.dropna(subset=['create_time_incident', 'neighborhood'])
    df = df.sort_values(by="create_time_incident", ascending=False)
    return df

df_crime = load_data()

# === Helper Functions ===
def clean_text(text):
    if not isinstance(text, str) or not text.strip(): return "Not Reported"
    text = text.upper()
    replacements = {
        "ARR:": "Arrest made", "ADV:": "Advised", "NTR:": "Nothing to report", "GOA:": "Gone on arrival",
        "CAN:": "Cancelled", "TC:": "Transferred call", "DIRPAT": "Directed Patrol", "MHC": "Mental Health Crisis",
        "SOW:": "Sent on way", "INV:": "Investigated", "NRBURG": "False Alarm", "REPO": "Towed Vehicle"
    }
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.strip().title()

def filter_rows(question, df):
    q = question.lower()
    filtered = df.copy()
    now = pd.Timestamp.now()

    if "today" in q:
        filtered = filtered[filtered['create_time_incident'].dt.date == now.date()]
    elif "yesterday" in q:
        filtered = filtered[filtered['create_time_incident'].dt.date == (now - pd.Timedelta(days=1)).date()]
    elif "this week" in q or "last week" in q:
        filtered = filtered[filtered['create_time_incident'] >= (now - pd.Timedelta(weeks=1))]
    elif "last month" in q:
        filtered = filtered[filtered['create_time_incident'] >= (now - pd.DateOffset(months=1))]
    elif "last year" in q:
        filtered = filtered[filtered['create_time_incident'] >= (now - pd.DateOffset(years=1))]

    match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december) (\d{4})", q)
    if match:
        month_str, year = match.groups()
        month_num = pd.to_datetime(month_str[:3], format='%b').month
        df['year'] = df['create_time_incident'].dt.year
        df['month'] = df['create_time_incident'].dt.month
        filtered = filtered[(df['year'] == int(year)) & (df['month'] == month_num)]

    for hood in df['neighborhood'].dropna().unique():
        if hood.lower() in q:
            filtered = filtered[filtered['neighborhood'] == hood]
            break

    keywords = ["shooting", "robbery", "patrol", "assault", "theft", "fire", "arrest", "accident", "traffic", "gun"]
    for kw in keywords:
        if kw in q:
            filtered = filtered[
                filtered['incident_type_desc'].str.contains(kw, case=False, na=False) |
                filtered['incident_type_id'].str.contains(kw, case=False, na=False)
            ]
            break

    return filtered.sort_values("create_time_incident", ascending=False).head(25)

def generate_answer(question, df):
    rows = filter_rows(question, df)
    if rows.empty:
        return "\ud83d\udeab No incidents found matching your question."

    valid_rows, ignored_rows = [], []
    for _, row in rows.iterrows():
        disp = str(row.get('disposition_text', '')).upper()
        if any(bad in disp for bad in ['CAN:', 'CANCEL', 'TC:', 'NTR:', 'USED CLEAR BUTTON']):
            ignored_rows.append(row)
        else:
            valid_rows.append(row)

    if not valid_rows:
        return f"\u26a0\ufe0f All {len(ignored_rows)} matched incidents were cancelled or administrative."
    if len(valid_rows) == 1:
        row = valid_rows[0]
        return f"Only one valid incident was found:\n\n\ud83d\udcc5 {row['create_time_incident'].strftime('%b %d, %Y')}\n\ud83d\udccd {clean_text(row.get('neighborhood'))}\n\ud83d\udcdd {clean_text(row.get('incident_type_desc'))}\n\ud83d\uded9 Outcome: {clean_text(row.get('disposition_text'))}\n\ud83d\udea8 Priority: {row.get('priority', 'N/A')}"

    context_lines = []
    incident_type_counts = Counter()
    for row in valid_rows:
        context_lines.append(f"- Date: {row['create_time_incident'].strftime('%b %d, %Y')}, Incident: {clean_text(row.get('incident_type_desc'))}, Neighborhood: {clean_text(row.get('neighborhood'))}, Priority: {row.get('priority', 'N/A')}, Outcome: {clean_text(row.get('disposition_text'))}")
        incident_type_counts[clean_text(row.get('incident_type_desc'))] += 1

    context = "\n".join(context_lines)
    incident_summary = ", ".join(f"{c} {t.lower() + ('s' if c > 1 else '')}" for t, c in incident_type_counts.items())

    prompt = f"""
Citizen asked: \"{question}\"
Out of {len(valid_rows) + len(ignored_rows)} total incidents, {len(ignored_rows)} were excluded. The remaining {len(valid_rows)} included: {incident_summary}.
{context}
Summarize what happened in a helpful and human-friendly paragraph:
"""

    return summarizer(prompt, max_length=300, truncation=True)[0]['generated_text']

# === Suggested Questions ===
sample_questions = [
    "What crimes happened in Over-the-Rhine last week?",
    "Any arrests in Walnut Hills in March 2025?",
    "Were there any shootings in Avondale this year?",
    "Tell me about police activity in Westwood yesterday.",
    "What kind of incidents happened in downtown Cincinnati in February 2024?"
]

st.markdown("**\ud83d\udccc Try a suggested question:**")
cols = st.columns(len(sample_questions))
for i, q in enumerate(sample_questions):
    if cols[i].button(q):
        st.session_state["preset"] = q

question = st.text_input("Ask a question:", value=st.session_state.get("preset", ""))

if st.button("Submit"):
    with st.spinner("\ud83e\udde0 Bot is thinking..."):
        response = generate_answer(question, df_crime)
        st.markdown("---")
        st.markdown(f"### \ud83d\udcec Response:\n{response}")
