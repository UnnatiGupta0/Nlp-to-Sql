# 🧠 NLP to SQL Converter

This project fine-tunes a T5 model to convert English questions into SQL queries using a custom employee dataset.

## 📁 Files Included

- `nlp_to_sql.ipynb` – Colab notebook for training and testing the model
- `company_data_dirty.json` – Sample employee dataset in JSON format
- `t5_sql_model/` – Trained T5 model folder (saved locally)
- `app.py` – Terminal app to test the model offline
- `README.md` – This file

---

## 🚀 How to Run the App Locally (on your Mac)

### 1. Install dependencies

Open Terminal and run:

```bash
pip install transformers torch pandas
