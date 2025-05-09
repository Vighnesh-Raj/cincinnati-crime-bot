# === STREAMLIT: CINCINNATI CRIME CHATBOT ===
import pandas as pd
import re
import mlflow
import streamlit as st
import requests
from collections import Counter
from huggingface_hub import hf_hub_download

# === CONFIG ===
BACKEND_URL = "https://96dd-35-229-67-215.ngrok-free.app/generate"  # <-- Your Colab public URL

st.set_page_config(page_title="Cincinnati Crime Chatbot", page_icon="🚓")
st.title("🚔 Cincinnati Crime Chatbot")
st.markdown("Ask about recent police activity in your neighborhood.")

mlflow.set_tracking_uri("http://3.145.180.106:5000")
mlflow.set_experiment("Cincinnati Crime Chatbot")

# === Load Dataset ===
@st.cache_data(show_spinner="📂 Loading police reports...")
def load_data():
    try:
        st.info("🔄 Downloading dataset from Hugging Face...")
        path = hf_hub_download(
            repo_id="mlsystemsg1/cincinnati-crime-data",
            repo_type="dataset",
            filename="calls_for_service_latest.csv"
        )
        st.success("✅ Data downloaded!")

        st.info("🧼 Cleaning data...")
        df = pd.read_csv(path, low_memory=False)
        df['incident_type_desc'] = df['incident_type_desc'].fillna(df['incident_type_id'])
        df['priority'] = pd.to_numeric(df['priority'], errors='coerce')
        df['create_time_incident'] = pd.to_datetime(df['create_time_incident'], errors='coerce')
        for col in ['sna_neighborhood', 'cpd_neighborhood']:
            df[col] = df[col].astype(str).str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)
        df['neighborhood'] = df['cpd_neighborhood'].combine_first(df['sna_neighborhood'])

        df = df.dropna(subset=['create_time_incident', 'neighborhood']).sort_values("create_time_incident", ascending=False)
        st.success("✅ Data cleaned and ready!")
        return df
    except Exception as e:
        st.error(f"🚨 Failed to load or process data: {str(e)}")
        raise

df = load_data()

# === Helpers ===
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return "Not Reported"
    text = text.upper()
    replacements = {
        "ARR:": "Arrest made", "ADV:": "Advised", "NTR:": "Nothing to report",
        "GOA:": "Gone on arrival", "CAN:": "Cancelled", "TC:": "Transferred call",
        "DIRPAT": "Directed Patrol", "MHC": "Mental Health Crisis", "SOW:": "Sent on way",
        "INV:": "Investigated", "NRBURG": "False Alarm", "REPO": "Towed Vehicle"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip().title()

def get_relevant_rows(question, df):
    q = question.lower()
    filtered = df.copy()
    now = pd.Timestamp.now()
    st.info("🔍 Filtering data based on user query...")

    try:
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

        st.success(f"✅ Filtered {len(filtered)} matching incidents.")
    except Exception as e:
        st.error(f"❌ Error filtering data: {str(e)}")
        return pd.DataFrame()

    return filtered.sort_values("create_time_incident", ascending=False).head(25)

# === Call Remote LLM ===
def call_colab_llm(prompt: str):
    try:
        res = requests.post(BACKEND_URL, json={"question": prompt}, timeout=30)
        res.raise_for_status()
        return res.json()["response"]
    except Exception as e:
        return f"❌ Error calling LLM backend: {str(e)}"

# === Final Inference ===
def answer_with_llm(question, df):
    try:
        valid_rows = []
        ignored_rows = []
        for _, row in df.iterrows():
            disp = str(row.get('disposition_text', '')).upper()
            if any(bad in disp for bad in ['CAN:', 'CANCEL', 'TC:', 'NTR:', 'USED CLEAR BUTTON']):
                ignored_rows.append(row)
            else:
                valid_rows.append(row)

        if not valid_rows:
            return f"⚠️ All {len(ignored_rows)} matched incidents were cancelled or administrative.", None
        if len(valid_rows) == 1:
            row = valid_rows[0]
            return f"📅 {row['create_time_incident'].strftime('%b %d, %Y')}, 🏙️ {clean_text(row['neighborhood'])}, 📋 {clean_text(row['incident_type_desc'])}, 🔚 {clean_text(row['disposition_text'])}, 🚨 Priority: {row.get('priority', 'N/A')}", None

        context_lines = []
        incident_type_counts = Counter()
        for row in valid_rows:
            context_lines.append(f"- {row['create_time_incident'].strftime('%b %d')}: {clean_text(row['incident_type_desc'])} in {clean_text(row['neighborhood'])}, priority {row.get('priority', 'N/A')}")
            incident_type_counts[clean_text(row['incident_type_desc'])] += 1

        incident_summary = ", ".join(f"{c} {t.lower() + ('s' if c > 1 else '')}" for t, c in incident_type_counts.items())
        context = "\n".join(context_lines)
        prompt = f"""
Citizen asked: \"{question}\"
Out of {len(valid_rows) + len(ignored_rows)} total incidents, {len(ignored_rows)} were excluded. The remaining {len(valid_rows)} included: {incident_summary}.
{context}
Summarize what happened in a helpful and human-friendly paragraph:
""".strip()

        with mlflow.start_run() as run:
            mlflow.log_param("question", question)
            mlflow.log_metric("num_valid_incidents", len(valid_rows))
            response = call_colab_llm(prompt)
            mlflow.log_text(response, "summary.txt")
            return response, run.info.run_id
    except Exception as e:
        return f"❌ Summary generation failed: {str(e)}", None

# === UI ===
question = st.text_input("Ask your question:").strip()
if question:
    with st.spinner("🔍 Analyzing your request..."):
        try:
            filtered = get_relevant_rows(question, df)

            with st.expander("🔎 Filtered Data Preview"):
                st.dataframe(filtered[['create_time_incident', 'incident_type_id', 'cpd_neighborhood', 'priority']].head(10))

            response, run_id = answer_with_llm(question, filtered)
            st.success("✅ Done!")
            st.markdown("### 🤖 Response:")
            st.write(response)

            st.markdown("### 🗳️ Was this helpful?")
            col1, col2 = st.columns(2)
            if col1.button("👍 Yes"):
                mlflow.set_tag("feedback", "thumbs_up")
                st.success("Thanks for your feedback!")
            if col2.button("👎 No"):
                mlflow.set_tag("feedback", "thumbs_down")
                st.warning("Thanks for letting us know!")
        except Exception as e:
            st.error(f"🚫 App crashed: {str(e)}")
