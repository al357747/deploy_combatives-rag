

import gradio as gr
from rag_core import answer_question

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
        fn=answer_question,
        inputs=[question, top_k, discipline, show_sources, temperature],
        outputs=[answer, sources_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)