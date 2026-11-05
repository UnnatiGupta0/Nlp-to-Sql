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
| Baseline | 1,000 | _run to fill_ | | | | |
| Improved | 2,000 | _run to fill_ | | | | |

## Run it

```bash
pip install -r requirements.txt
python train.py          # ~3 min on a Colab T4 GPU, ~25 min on a laptop CPU
python app.py            # terminal
python gradio_app.py     # web demo at http://127.0.0.1:7860
```

Or open `notebooks/nlp_to_sql_v2.ipynb` in Google Colab and Run all.

## Limitations

- Templated questions: the model is strong on these patterns but will struggle with phrasings far from the training data.
- One fixed schema; a new database would need new training pairs. For many schemas, prompting a large LLM with the schema is the more flexible choice; a small fine-tuned model wins on cost, latency and privacy for one fixed schema.
