from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import json

model = T5ForConditionalGeneration.from_pretrained("t5_sql_model")
tokenizer = T5Tokenizer.from_pretrained("t5_sql_model")

with open("company_data_dirty.json") as f:
    data = json.load(f)

df = pd.DataFrame(data["Emp"])
df = df.replace("N/A", pd.NA).dropna()
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["doj"] = pd.to_datetime(df["doj"], errors="coerce")
df = df.dropna().reset_index(drop=True)

def generate_sql(question):
    input_text = "translate English to SQL: " + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

while True:
    q = input("\nAsk your question (or type 'exit'): ")
    if q.lower() == "exit":
        break
    sql = generate_sql(q)
    print("Generated SQL:", sql)
