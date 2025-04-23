# STEP 1: Install requirements
!pip install -q streamlit transformers huggingface_hub pyngrok pandas fuzzywuzzy[speedup] mlflow
!ngrok config add-authtoken 2w5pD0aBQWixs2qmoCJ7QpX6B37_7XHw93dLnGpqSNxXCytj8

# STEP 2: Write app.py
from pathlib import Path

app_code = """
import streamlit as st
import pandas as pd
import calendar
import re
import time
import mlflow
from datetime import timedelta
from fuzzywuzzy import fuzz, process
from transformers import pipeline
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Cincinnati Crime Chatbot", page_icon="🚔")

mlflow.set_tracking_uri("http://3.145.180.106:5000")
mlflow.set_experiment("Cincinnati Crime Chatbot")

@st.cache_data
def load_crime_data():
    csv_path = hf_hub_download(
        repo_id="mlsystemsg1/cincinnati-crime-data",
        repo_type="dataset",
        filename="calls_for_service_latest.csv"
    )
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [col.lower() for col in df.columns]
    df.dropna(subset=['cpd_neighborhood'], inplace=True)
    df['create_time_incident'] = pd.to_datetime(df['create_time_incident'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

OFFENSE_GROUPS = {
    "robbery": ["ROBBERY PERSONAL (JO)(W)", "ROBBERY PERSONAL (IP)", "ROBBERY BUSINESS (NIP)"],
    "assault": ["ASSAULT (IP)(W)", "ASSAULT (JO)(W)(E)", "SEX ASSAULT ADULT (S)",
                "SEX ASSAULT CHILD (IP)", "SEX ASSAULT CHILD (S)(W)(E)", "RAPER"],
    "disturbance": ["DISTURBANCE VERB (IP)(W)", "DISTURBANCE VERB (S)", "DISTURBANCE PHYS (JO)",
                    "DISTURBANCE VERB (S)(W)", "DISTURBANCE PHYS (S)(E)",
                    "FAMILY DIST PHYS (JO)(E)", "FAMILY DIST PHYS (S)", "FAMILY DIST UNKN (JO)"],
    "theft": ["VEHICLE THEFT (JO)(W)", "U-AUTO THEFT/RECOVERY RPT",
              "THEFT (IP)(W)", "OCR THEFT ATTEMPT (NIP)"],
    "drug": ["DRUG USE/POSSESS (IP)(W)(E)", "DRUG SALE (IP)(W)(E)", "ADV - DRUG SALE (NIP)"]
}

def get_relevant_rows(question, df):
    q = question.lower()
    filtered = df.copy()

    neighborhoods = df['cpd_neighborhood'].dropna().unique()
    matched_hood = next((hood for hood in neighborhoods if hood.lower() in q), None)
    if matched_hood:
        filtered = filtered[filtered['cpd_neighborhood'].str.lower() == matched_hood.lower()]

    for group, values in OFFENSE_GROUPS.items():
        if group in q:
            filtered = filtered[filtered['incident_type_id'].isin(values)]
            break
    else:
        offenses = df['incident_type_id'].dropna().astype(str).unique()
        match, score = process.extractOne(q, offenses, scorer=fuzz.partial_ratio)
        if score >= 85:
            filtered = filtered[filtered['incident_type_id'].str.lower() == match.lower()]

    now = pd.Timestamp.now()
    if match := re.search(r"(20\\d{2})[-/](\\d{1,2})[-/](\\d{1,2})", q):
        y, m, d = map(int, match.groups())
        filtered = filtered[filtered['create_time_incident'].dt.date == pd.Timestamp(y, m, d).date()]
    elif "last week" in q or "past week" in q:
        filtered = filtered[filtered['create_time_incident'] >= now - timedelta(days=7)]
    elif "last month" in q or "past month" in q:
        filtered = filtered[filtered['create_time_incident'] >= now - timedelta(days=30)]
    elif "yesterday" in q:
        filtered = filtered[filtered['create_time_incident'].dt.date == (now - timedelta(days=1)).date()]
    elif "today" in q:
        filtered = filtered[filtered['create_time_incident'].dt.date == now.date()]

    filtered = filtered.dropna(subset=['create_time_incident'])
    filtered = filtered.sort_values(by='create_time_incident', ascending=False)
    return filtered

def answer_with_llm(question, data_rows, model):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("question", question)
        mlflow.log_param("match_count", len(data_rows))

        if re.search(r"(last|most recent|latest)", question.lower()) and not data_rows.empty:
            data_rows = data_rows.sort_values(by='create_time_incident', ascending=False).head(1)

        if len(data_rows) == 1:
            row = data_rows.iloc[0]
            date = pd.to_datetime(row['create_time_incident']).date()
            offense = row['incident_type_id']
            hood = row['cpd_neighborhood']
            priority = row['priority']
            answer = f"{date}, a {offense} (Priority {priority}) occurred in {hood}"
            mlflow.log_text(answer, "answer.txt")
            return answer, run_id

        context = "\\n".join([
            f"On {pd.to_datetime(r['create_time_incident']).date()}, a {r['incident_type_id']} (Priority {r['priority']}) occurred in {r['cpd_neighborhood']} (Incident #{r['event_number']})."
            for _, r in data_rows.head(5).iterrows()
        ])

        prompt = f\"\"\"\nYou are a helpful assistant analyzing Cincinnati crime data.\n\nHere are some relevant data points:\n{context}\n\nNow answer this question based on the above:\n{question}\n\"\"\".strip()

        mlflow.log_text(prompt, "prompt.txt")
        result = model(prompt, max_new_tokens=150)[0]['generated_text']
        response = result.strip()
        mlflow.log_text(response, "answer.txt")
        return response, run_id

# --- Streamlit App ---
st.title("🔍 Cincinnati Crime Chatbot")
st.markdown("Ask things like 'When was the last robbery in Clifton?' or 'What is the safest neighborhood?'")

df = load_crime_data()
model = load_model()

question = st.text_input("Ask your question:").strip()

if question:
    with st.spinner("Analyzing..."):
        filtered = get_relevant_rows(question, df)

        with st.expander("🔍 Filtered Data Preview"):
            st.dataframe(filtered[['create_time_incident', 'incident_type_id', 'cpd_neighborhood', 'priority']].head(10))

        response, run_id = answer_with_llm(question, filtered, model)
        st.success("Done!")
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
"""

Path("app.py").write_text(app_code, encoding="utf-8", errors="surrogatepass")

# STEP 3: Run and expose
from pyngrok import ngrok
import threading
import time
import os

ngrok.kill()

def run_app():
    os.system("streamlit run app.py --server.enableCORS false")

thread = threading.Thread(target=run_app)
thread.start()
time.sleep(6)

public_url = ngrok.connect("http://localhost:8501")
print(f"🌐 Your Streamlit app is live at: {public_url}")
