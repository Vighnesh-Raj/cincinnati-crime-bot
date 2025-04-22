# app.py (Gradio frontend to Colab-hosted FLAN-T5)
import gradio as gr
import requests

# Replace with your current ngrok URL
BACKEND_URL = "https://96dd-35-229-67-215.ngrok-free.app/generate"

def ask_bot(question):
    try:
        response = requests.post(BACKEND_URL, json={"question": question}, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "No response from model.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

iface = gr.Interface(
    fn=ask_bot,
    inputs=gr.Textbox(lines=2, placeholder="Ask about crime in your neighborhood..."),
    outputs="text",
    title="Cincinnati Crime Chatbot 🚓",
    description="This chatbot summarizes recent police incidents in Cincinnati using FLAN-T5-Large hosted on Google Colab."
)

if __name__ == "__main__":
    iface.launch()
