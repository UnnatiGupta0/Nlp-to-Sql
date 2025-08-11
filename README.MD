# NLP to SQL Converter

A simple yet functional project where I fine-tuned a T5 model to translate everyday English questions into SQL queries.  
It’s built on a custom employee dataset and can be run both on Google Colab and locally through the terminal.


## 📁 Files Included

- `nlp_to_sql.ipynb` – Colab notebook for training and testing the model
- `company_data_dirty.json` – Sample employee dataset in JSON format
- `t5_sql_model/` – Trained T5 model folder (saved locally)
- `app.py` – Terminal app to test the model offline
- `README.md` – This file

---
## Demo

![Demo Screenshot](demo%20ss.jpeg)


##  How to Run the App Locally (on your Mac)

### 1. Install dependencies

Open Terminal and run:

```bash
pip install transformers torch pandas

### 2. Run the App
```bash
python app.py

Type an English question (e.g. Show all employees in the Sales department).
The model will output the corresponding SQL query.


