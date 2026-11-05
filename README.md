# NLP to SQL — fine-tuned T5

Turns plain-English business questions into SQL that runs on a 5-table company database, and shows how **targeted training data** fixes the query types a model fails on.

```
"Which male employees work on Python projects?"
→ SELECT DISTINCT employees.name FROM employees JOIN emp_project ON employees.id = emp_project.empid
  WHERE employees.gender = 'Male' AND emp_project.tech_stack = 'Python'
```

## What's in here

| File | What it does |
|---|---|
| `company_data_dirty.json` | Raw data: employees, projects, emp_project, milestones, proj_delivery (with `N/A`s and messy dates) |
| `nlp2sql.py` | Cleans the data into SQLite, generates question→SQL pairs, scores predictions |
| `train.py` | Runs the full experiment (baseline vs. improved) and saves the model + results |
| `app.py` | Terminal app: ask a question, see the SQL and the rows it returns |
| `gradio_app.py` | Web demo of the same thing |
| `notebooks/nlp_to_sql_v2.ipynb` | Colab walkthrough of the experiment |
| `notebooks/v1_original.ipynb` | First version: single-table queries only, no evaluation |

## Method

1. **Data:** cleaned the dirty JSON (dropped `N/A` rows, normalised dates) and loaded all 5 tables into SQLite.
2. **Pairs:** generated question→SQL pairs from templates with varied phrasings:
   - *simple*: SELECT / WHERE / COUNT on one table (what v1 trained on)
   - *aggregation*: GROUP BY, COUNT(DISTINCT)
   - *join*: 2- and 3-table JOINs
   - *subquery*: NOT IN, IN + GROUP BY … HAVING
3. **Held-out test set:** 300 questions (150 simple, 150 complex) set aside **before** training; the script asserts none appear in training data.
4. **Baseline:** fine-tuned `t5-small` on 1,000 simple pairs only.
5. **Error analysis:** broke baseline failures down by query type.
6. **Improved:** added 1,000 pairs targeting aggregations, joins and subqueries; retrained from scratch.
7. **Metric:** **execution accuracy** (run the predicted SQL and the reference SQL, compare the rows returned), plus exact match.

## Results

Run `python train.py` to reproduce; numbers are written to `results/results.json`.

| Model | Train pairs | Execution accuracy | Simple | Aggregation | Join | Subquery |
|---|---|---|---|---|---|---|
| Baseline | 1,000 | **45.7%** | 91.3% | 0% | 0% | 0% |
| Improved | 2,000 | **94.7%** | 89.3% | 100% | 100% | 100% |

Held-out test set: 300 questions (150 simple, 150 complex). t5-small, 8 epochs, Colab T4 GPU (training: 1.0 min baseline, 2.4 min improved).

**Error analysis.** The baseline got 0% on every complex category: it had never seen JOINs, GROUP BY or subqueries, so it forced them into single-table queries (e.g. `COUNT(*) FROM employees WHERE doj > 'React'` for a tech-stack question). Its simple-query misses were name-copying errors (`'Sandra'` → `'Suzan'`). Adding targeted pairs fixed all three complex categories, at a small cost on simple queries (91.3% → 89.3%, 3 of 150).

## Run it

**Quick demo, no training needed.** The trained model is on Hugging Face: [Unnz/nlp-to-sql-t5](https://huggingface.co/Unnz/nlp-to-sql-t5)

```bash
pip install -r requirements.txt
MODEL_ID=Unnz/nlp-to-sql-t5 python gradio_app.py   # web demo at http://127.0.0.1:7860
```

**Full experiment:**

```bash
pip install -r requirements.txt
python train.py          # ~3 min on a Colab T4 GPU, ~25 min on a laptop CPU
python app.py            # terminal
python gradio_app.py     # web demo at http://127.0.0.1:7860
```

Or open `notebooks/nlp_to_sql_v2.ipynb` in Google Colab and Run all.

## Limitations

- **In-distribution test.** Test questions are held out, but they come from the same templates as training, so 100% on complex categories shows the model learned those patterns, not that it handles any phrasing. Execution accuracy equals exact match for both models for the same reason.
- **Unseen values.** The model copies values it saw in training well, but not new ones: *"How many female employees are from India?"* produced `country = 'Inde'`, because India (3 employees) was below the 4-employee cut-off used for templates. Fixes: include every value in training, use a bigger model, or constrain generation to values that exist in the database.
- The data is synthetic (Faker), so names and genders don't always match; query results can look odd even when the SQL is right.
- One fixed schema; a new database would need new training pairs. For many schemas, prompting a large LLM with the schema is the more flexible choice; a small fine-tuned model wins on cost, latency and privacy for one fixed schema.
