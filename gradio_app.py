"""Web demo: type a question, get SQL and the query result.

Local:   python gradio_app.py            (uses ./t5_sql_model from train.py)
Hosted:  set MODEL_ID to a Hugging Face model repo, e.g. UnnatiGupta0/nlp-to-sql-t5
"""
import os

import gradio as gr
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

from nlp2sql import load_db

MODEL_ID = os.environ.get("MODEL_ID", "t5_sql_model")
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
conn, tables = load_db()


def answer(question):
    ids = tokenizer("translate English to SQL: " + question, return_tensors="pt")
    sql = tokenizer.decode(model.generate(**ids, max_new_tokens=128)[0], skip_special_tokens=True)
    try:
        cur = conn.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        return sql, df.head(50), f"{len(df)} row(s)"
    except Exception as e:
        return sql, pd.DataFrame(), f"Query failed to run: {e}"


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Ask a question about the company database"),
    outputs=[gr.Code(label="Generated SQL", language="sql"), gr.Dataframe(label="Result"), gr.Textbox(label="Status")],
    title="NLP to SQL (fine-tuned T5-small)",
    description="Tables: employees, projects, emp_project, milestones, proj_delivery.",
    examples=["How many female employees are from Thailand?", "Which male employees work on Python projects?",
              "Show employees from Thailand working on more than 2 projects", "How many projects use Java?"],
)

if __name__ == "__main__":
    demo.launch()
