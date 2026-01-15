import os
import requests
import gradio as gr

API_BASE = os.environ.get("API_BASE")
if not API_BASE:
    raise RuntimeError("Missing required env var API_BASE (e.g., https://deploy-combatives-rag.onrender.com)")

def ask_api(question, top_k, discipline, show_sources, temperature):
    payload = {
        "question": question,
        "top_k": top_k,
        "discipline_filter": discipline,
        "show_sources": show_sources,
        "temperature": temperature,
    }

    r = requests.post(
        f"{API_BASE}/query",
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()

    return data.get("answer", ""), data.get("sources", "")

with gr.Blocks(title="Combatives RAG") as demo:
    gr.Markdown("# Combatives RAG\nAsk questions over your Muay Thai / Jiu Jitsu notes.")

    with gr.Row():
        question = gr.Textbox(
            label="Question",
            placeholder="e.g., How do I use teeps to maintain forward pressure?",
            lines=3,
        )

    with gr.Row():
        discipline = gr.Dropdown(
            choices=["all", "muay-thai", "jiu-jitsu"],
            value="all",
            label="Discipline filter",
        )
        top_k = gr.Slider(1, 10, value=3, step=1, label="Top-K retrieved chunks")
        temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Temperature")

    show_sources = gr.Checkbox(value=True, label="Show retrieval sources")
    run_btn = gr.Button("Ask")

    answer = gr.Textbox(label="Answer", lines=12)
    sources_box = gr.Textbox(label="Sources (retrieval)", lines=6)

    run_btn.click(
        fn=ask_api,
        inputs=[question, top_k, discipline, show_sources, temperature],
        outputs=[answer, sources_box],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)