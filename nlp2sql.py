"""Shared helpers: load the company database, generate question->SQL pairs, score predictions.

Database: 5 tables from company_data_dirty.json, cleaned and loaded into in-memory SQLite.
  employees(id, name, dob, doj, gender, city, country)
  projects(id, name, tech_stack, start_date, end_date, project_mgr)
  emp_project(prjid, empid, startdate, end_date, tech_stack)
  milestones(id, milestone_desc)
  proj_delivery(prj_id, milestone_id, deliverables)
"""
import json
import random
import re
import sqlite3
from collections import Counter

import pandas as pd

DATA_FILE = "company_data_dirty.json"


# ---------------------------------------------------------------- database
def load_db(path=DATA_FILE):
    """Clean the dirty JSON (drop 'N/A' rows, normalise dates) and load it into SQLite."""
    with open(path) as f:
        raw = json.load(f)

    def clean(rows, date_cols=()):
        df = pd.DataFrame(rows).replace("N/A", pd.NA).dropna()
        for c in date_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
        return df.dropna().reset_index(drop=True)

    tables = {
        "employees": clean(raw["Emp"], ["dob", "doj"]),
        "projects": clean(raw["Project"], ["projects_start_date", "project_end_date"]).rename(
            columns={"projects_start_date": "start_date", "project_end_date": "end_date"}),
        "emp_project": clean(raw["Emp_Project"], ["startdate", "end_date"]),
        "milestones": clean(raw["Milestone"]),
        "proj_delivery": clean(raw["Proj_Delivery"]),
    }
    conn = sqlite3.connect(":memory:")
    for name, df in tables.items():
        df.to_sql(name, conn, index=False)
    return conn, tables


# ---------------------------------------------------------------- data generation
def _values(tables):
    emp, prj = tables["employees"], tables["projects"]
    vc = emp["country"].value_counts()
    return {
        "genders": ["Male", "Female"],
        "countries": sorted(vc[vc >= 4].index),
        "first_names": sorted({n.split()[0] for n in emp["name"]}),
        "techs": sorted(set(prj["tech_stack"]) | set(tables["emp_project"]["tech_stack"])),
        "managers": sorted(set(prj["project_mgr"])),
    }


def _pick(*phrasings):
    return random.choice(phrasings)


def simple_pair(v):
    """The single-table templates from the original version (SELECT / WHERE / COUNT)."""
    g, c = random.choice(v["genders"]), random.choice(v["countries"])
    t = random.choice(["gender_country", "dob_before", "doj_after", "count_gender", "name_match"])
    if t == "gender_country":
        q = _pick(f"Show all {g.lower()} employees from {c}", f"List {g.lower()} employees in {c}",
                  f"Get every {g.lower()} employee based in {c}")
        s = f"SELECT * FROM employees WHERE gender = '{g}' AND country = '{c}'"
    elif t == "dob_before":
        y = random.randint(1960, 2005)
        q = _pick(f"List employees born before {y}", f"Show employees born before {y}",
                  f"Which employees were born before {y}?")
        s = f"SELECT * FROM employees WHERE dob < '{y}-01-01'"
    elif t == "doj_after":
        y = random.randint(2010, 2024)
        q = _pick(f"Show employees who joined after {y}", f"List employees who joined after {y}",
                  f"Which employees joined after {y}?")
        s = f"SELECT * FROM employees WHERE doj > '{y}-12-31'"
    elif t == "count_gender":
        q = _pick(f"How many {g.lower()} employees are from {c}?", f"Count the {g.lower()} employees in {c}",
                  f"Number of {g.lower()} employees in {c}")
        s = f"SELECT COUNT(*) FROM employees WHERE gender = '{g}' AND country = '{c}'"
    else:
        n = random.choice(v["first_names"])
        q = _pick(f"Find employees named {n}", f"Show employees whose first name is {n}")
        s = f"SELECT * FROM employees WHERE name LIKE '{n} %'"
    return q, s, "simple"


def complex_pair(v):
    """Harder patterns the original model never saw: GROUP BY, JOINs, subqueries, HAVING."""
    g, c = random.choice(v["genders"]), random.choice(v["countries"])
    tech, m = random.choice(v["techs"]), random.choice(v["managers"])
    y, n = random.randint(2010, 2023), random.randint(1, 3)
    t = random.choice(["group_by", "count_distinct", "count_mgr", "join_tech", "join_mgr",
                       "not_in_country", "not_in_tech", "having"])
    if t == "group_by":
        q = _pick(f"How many {g.lower()} employees joined after {y} in each country?",
                  f"Count {g.lower()} employees who joined after {y} per country",
                  f"Number of {g.lower()} employees joined after {y} by country")
        s = f"SELECT country, COUNT(*) FROM employees WHERE gender = '{g}' AND doj > '{y}-12-31' GROUP BY country"
        kind = "aggregation"
    elif t == "count_distinct":
        q = _pick(f"How many employees work on {tech} projects?", f"Count employees assigned to {tech} work",
                  f"Number of employees working with {tech}")
        s = f"SELECT COUNT(DISTINCT empid) FROM emp_project WHERE tech_stack = '{tech}'"
        kind = "aggregation"
    elif t == "count_mgr":
        q = _pick(f"How many projects does {m} manage?", f"Count the projects managed by {m}",
                  f"Number of projects under {m}")
        s = f"SELECT COUNT(*) FROM projects WHERE project_mgr = '{m}'"
        kind = "aggregation"
    elif t == "join_tech":
        q = _pick(f"List {g.lower()} employees working on {tech} projects",
                  f"Which {g.lower()} employees are assigned to {tech} work?",
                  f"Show {g.lower()} employees who work with {tech}")
        s = ("SELECT DISTINCT employees.name FROM employees JOIN emp_project ON employees.id = emp_project.empid "
             f"WHERE employees.gender = '{g}' AND emp_project.tech_stack = '{tech}'")
        kind = "join"
    elif t == "join_mgr":
        q = _pick(f"Which {g.lower()} employees work on projects managed by {m}?",
                  f"List {g.lower()} employees on {m}'s projects",
                  f"Show {g.lower()} employees in projects run by {m}")
        s = ("SELECT DISTINCT employees.name FROM employees JOIN emp_project ON employees.id = emp_project.empid "
             f"JOIN projects ON emp_project.prjid = projects.id WHERE employees.gender = '{g}' "
             f"AND projects.project_mgr = '{m}'")
        kind = "join"
    elif t == "not_in_country":
        q = _pick(f"Find employees from {c} who are not assigned to any project",
                  f"Show {c} employees with no project", f"Which employees in {c} have no project?")
        s = f"SELECT * FROM employees WHERE country = '{c}' AND id NOT IN (SELECT empid FROM emp_project)"
        kind = "subquery"
    elif t == "not_in_tech":
        q = _pick(f"Show {g.lower()} employees who joined after {y} and never worked on {tech}",
                  f"List {g.lower()} employees joined after {y} with no {tech} project")
        s = (f"SELECT * FROM employees WHERE gender = '{g}' AND doj > '{y}-12-31' AND id NOT IN "
             f"(SELECT empid FROM emp_project WHERE tech_stack = '{tech}')")
        kind = "subquery"
    else:
        q = _pick(f"Show employees from {c} working on more than {n} projects",
                  f"List {c} employees assigned to over {n} projects",
                  f"Which employees in {c} have more than {n} projects?")
        s = (f"SELECT * FROM employees WHERE country = '{c}' AND id IN "
             f"(SELECT empid FROM emp_project GROUP BY empid HAVING COUNT(*) > {n})")
        kind = "subquery"
    return q, s, kind


def make_pairs(conn, tables, n_simple, n_complex, seed):
    """Unique questions only; drop pairs whose reference answer is empty (an empty answer is too easy to match)."""
    random.seed(seed)
    v = _values(tables)
    rows, seen = [], set()
    for maker, n in ((simple_pair, n_simple), (complex_pair, n_complex)):
        got, tries = 0, 0
        while got < n and tries < n * 100:
            tries += 1
            q, s, k = maker(v)
            res = run_sql(conn, s)
            if q in seen or not res or res == Counter([(0,)]):
                continue
            seen.add(q)
            rows.append((q, s, k))
            got += 1
    return pd.DataFrame(rows, columns=["question", "sql", "category"])


# ---------------------------------------------------------------- scoring
def run_sql(conn, sql):
    try:
        return Counter(map(tuple, conn.execute(sql).fetchall()))
    except Exception:
        return None


def execution_match(conn, pred, gold):
    """Correct if the predicted query returns exactly the same rows as the reference query."""
    p = run_sql(conn, pred)
    return p is not None and p == run_sql(conn, gold)


def normalise(sql):
    """Lowercase and ignore spacing differences (e.g. 'dob<' vs 'dob <') for exact-match scoring."""
    sql = re.sub(r"\s*([<>=(),*])\s*", r" \1 ", sql.strip().rstrip(";"))
    return " ".join(sql.split()).lower()
