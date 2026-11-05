"""Ask a question in plain English -> get SQL -> see the result from the company database.

Run `python train.py` first (it saves the model to ./t5_sql_model), then `python app.py`.
"""
import os
import sys

from transformers import T5ForConditionalGeneration, T5Tokenizer

from nlp2sql import load_db

MODEL_DIR = "t5_sql_model"
if not os.path.isdir(MODEL_DIR):
    sys.exit("No trained model found. Run `python train.py` first.")

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
conn, _ = load_db()


def generate_sql(question):
    ids = tokenizer("translate English to SQL: " + question, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=128)
    return tokenizer.decode(out[0], skip_special_tokens=True)


print("Tables: employees, projects, emp_project, milestones, proj_delivery")
print("Try: 'How many female employees are from Thailand?' or 'Which male employees work on Python projects?'")
while True:
    q = input("\nAsk your question (or type 'exit'): ").strip()
    if q.lower() == "exit":
        break
    sql = generate_sql(q)
    print("Generated SQL:", sql)
    try:
        rows = conn.execute(sql).fetchall()
        print(f"{len(rows)} row(s)")
        for r in rows[:10]:
            print("  ", r)
    except Exception as e:
        print("Query failed to run:", e)
