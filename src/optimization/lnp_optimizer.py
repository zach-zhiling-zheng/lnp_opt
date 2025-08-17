import os
import re
import json
import random
import datetime as dt
import time
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skopt import Optimizer
from skopt.space import Categorical

from openai import OpenAI
from dotenv import load_dotenv

from edbo.plus.optimizer_botorch import EDBOplus

load_dotenv()

# Read the Excel file at import time only if present, else delay until needed
def _load_lnp_df(xlsx_path: str):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path, header=None, names=['I', 'H', 'C', 'P', 'Fluorescence'])
    return df

I_range = [f"I{i}" for i in range(1, 5+1)]
H_range = [f"H{i}" for i in range(1, 5+1)]
C_range = [f"C{i}" for i in range(1, 5+1)]
P_range = [f"P{i}" for i in range(1, 4+1)]
all_conditions = [(i, h, c, p) for i in I_range for h in H_range for c in C_range for p in P_range]

def random_suggest(n, history, iteration, total_iterations):
    tried = set(tuple(h[:4]) for h in history)
    pool = [cond for cond in all_conditions if cond not in tried]
    if n > len(pool):
        n = len(pool)
    return random.sample(pool, n)

def bo_suggest(n, history, iteration, total_iterations):
    if not history:
        return random_suggest(n, history, iteration, total_iterations)

    hist_df = pd.DataFrame(history, columns=['I','H','C','P','Fluorescence'])
    hist_df = hist_df.dropna(subset=['Fluorescence'])

    space = [
        Categorical([f"I{i}" for i in range(1, 6)], name="I"),
        Categorical([f"H{i}" for i in range(1, 6)], name="H"),
        Categorical([f"C{i}" for i in range(1, 6)], name="C"),
        Categorical([f"P{i}" for i in range(1, 4+1)], name="P"),
    ]

    opt = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI")

    for _, row in hist_df.iterrows():
        opt.tell([(row['I'], row['H'], row['C'], row['P'])], [-float(row['Fluorescence'])])

    suggestions = opt.ask(n_points=n)
    suggestions = [(str(a), str(b), str(c), str(d)) for (a,b,c,d) in suggestions]
    return suggestions

def _extract_json(text: str):
    """Find the first top-level JSON object in a string and parse it."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return None

def _openai_client(api_key: str = None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

def llm_suggest(n, history, iteration, total_iterations, model="gpt-4o"):
    if not history:
        return random_suggest(n, history, iteration, total_iterations)

    hist_list = [
        {"Condition": idx+1, "I": h[0], "H": h[1], "C": h[2], "P": h[3], "Fluorescent": h[4] if len(h)>4 else None}
        for idx, h in enumerate(history)
    ]
    history_json = json.dumps({"conditions": hist_list}, indent=2)

    prompt = f"""
You are running LNP experiments by combining 4 chemicals I-H-P-C in one pot, with the goal of maximizing Fluorescent measurement outcome where I is Cationic (Ionizable) Lipids, H is Helper (Phospholipid) Lipids, P is PEGylated Lipids, and C is Cholesterol.
I, H, C each have 5 choices and P has 4 choices, pick 1 from each category per condition.

Here is previous experiment history:
{history_json}

Return {n} new suggestions as JSON only:
{{
  "conditions": [
    {{"Condition": <int>, "I": "I1", "H": "H2", "C": "C3", "P": "P1"}},
    ...
  ]
}}
Ensure no duplicate of already tried conditions.
""".strip()

    client = _openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    txt = resp.choices[0].message.content or ""
    parsed = _extract_json(txt)
    if not parsed or "conditions" not in parsed:
        return random_suggest(n, history, iteration, total_iterations)

    out = []
    tried = set(tuple(h[:4]) for h in history)
    for c in parsed["conditions"]:
        cond = (c.get("I"), c.get("H"), c.get("C"), c.get("P"))
        if None in cond:
            continue
        if cond in tried or cond in out:
            continue
        out.append(cond)
        if len(out) >= n:
            break
    if len(out) < n:
        out.extend(random_suggest(n - len(out), history, iteration, total_iterations))
    return out

def reasoning_llm_suggest(n, history, iteration, total_iterations, model="o1-preview"):
    # Same as llm_suggest but with a different model name. Could add more steps if desired.
    return llm_suggest(n, history, iteration, total_iterations, model=model)

def delete_existing_optimization_files(prefix="my_optimization"):
    for path in glob.glob(f"{prefix}_round*.csv"):
        os.remove(path)
    for path in glob.glob(f"pred_{prefix}_round*.csv"):
        os.remove(path)

def edbo_suggest(n, history, iteration, total_iterations, filename="my_optimization.csv"):
    if iteration == 0:
        if os.path.exists(filename):
            os.remove(filename)
        reaction_components = {
            "I": [f"I{i}" for i in range(1, 6)],
            "H": [f"H{i}" for i in range(1, 6)],
            "C": [f"C{i}" for i in range(1, 6)],
            "P": [f"P{i}" for i in range(1, 5)],
        }
        EDBOplus().generate_reaction_scope(components=reaction_components, filename=filename, check_overwrite=False)
        EDBOplus().run(
            filename=filename,
            objectives=["Fluorescence"],
            objective_mode=["max"],
            batch=n,
            columns_features="all",
            init_sampling_method="seed",
        )
    else:
        df_edbo = pd.read_csv(filename)
        pending_rows = df_edbo[(df_edbo["Fluorescence"] == "PENDING") & (df_edbo["priority"] == 1)]
        if len(pending_rows) < n:
            print(f"Warning: found only {len(pending_rows)} pending rows with priority 1.")
        for idx, row in pending_rows.iterrows():
            cond = (row["I"], row["H"], row["C"], row["P"])
            fl = next((h[-1] for h in history if h[:-1] == cond), None)
            if fl is not None:
                df_edbo.at[idx, "Fluorescence"] = fl
            else:
                df_edbo.at[idx, "priority"] = 0
        df_edbo.to_csv(filename, index=False)
        EDBOplus().run(
            filename=filename,
            objectives=["Fluorescence"],
            objective_mode=["max"],
            batch=n,
            columns_features="all",
            init_sampling_method="cvtsampling",
        )

    df_edbo = pd.read_csv(filename)
    out = []
    for _, row in df_edbo[df_edbo["priority"] == 1].head(n).iterrows():
        out.append((row["I"], row["H"], row["C"], row["P"]))
    return out

def execute(conditions, df):
    results = []
    for cond in conditions:
        row = df.loc[
            (df["I"] == cond[0]) & (df["H"] == cond[1]) & (df["C"] == cond[2]) & (df["P"] == cond[3])
        ]
        if not row.empty:
            fl = int(row["Fluorescence"].values[0])
            results.append(cond + (fl,))
        else:
            print(f"Condition not found in Excel: I={cond[0]}, H={cond[1]}, C={cond[2]}, P={cond[3]}")
            results.append(cond + (None,))
    return results

def run_experiment(n, iterations, repeat, suggest_func, method_name, df):
    all_histories = []
    for _ in range(repeat):
        history = []
        for i in range(iterations):
            suggestions = suggest_func(n, history, i, iterations)
            results = execute(suggestions, df)
            history.extend(tuple(r) for r in results)
        all_histories.append(history)
    return {method_name: all_histories}

def save_csv(results, filename, n, iterations, repeat):
    base = filename[:-4] if filename.endswith(".csv") else filename
    counter = 1
    while os.path.exists(f"{base}.csv") or os.path.exists(f"{base}.png"):
        base = f"{base}_{counter}"
        counter += 1

    csv_filename = f"{base}.csv"
    with open(csv_filename, "w") as f:
        f.write("Suggest_Method,n,Repeat,Iteration,Index,I,H,C,P,Fluorescence\n")
        for method, histories in results.items():
            for rpt, history in enumerate(histories, start=1):
                for idx, item in enumerate(history):
                    cond, fl = item[:-1], item[-1]
                    iter_num = idx // n + 1
                    i, h, c, p = cond
                    f.write(f"{method},{n},{rpt},{iter_num},{idx},{i},{h},{c},{p},{fl}\n")
    return csv_filename

def load_and_plot_summary(filename):
    df = pd.read_csv(filename)
    for method in df["Suggest_Method"].unique():
        mdf = df[df["Suggest_Method"] == method]
        iterations = int(mdf["Iteration"].max())
        repeats = int(mdf["Repeat"].max())
        ys = []
        for rpt in range(1, repeats+1):
            rdf = mdf[mdf["Repeat"] == rpt]
            max_so_far = -np.inf
            y = []
            for it in range(1, iterations+1):
                cur = rdf[rdf["Iteration"] == it]["Fluorescence"]
                cur = pd.to_numeric(cur, errors="coerce").dropna()
                if cur.empty:
                    y.append(max_so_far if np.isfinite(max_so_far) else np.nan)
                    continue
                m = cur.max()
                if m > max_so_far:
                    max_so_far = m
                y.append(max_so_far)
            ys.append(y)
        x = list(range(1, iterations+1))
        y_mean = np.nanmean(np.asarray(ys, dtype=float), axis=0)
        y_std = np.nanstd(np.asarray(ys, dtype=float), axis=0)
        plt.plot(x, y_mean, label=method)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel("Highest Observed Fluorescence")
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_and_plot_results(filename):
    df = pd.read_csv(filename)
    df["Fluorescence"] = pd.to_numeric(df["Fluorescence"], errors="coerce")
    methods = df["Suggest_Method"].unique()

    plt.figure(figsize=(10, 6))
    for method in methods:
        mdf = df[df["Suggest_Method"] == method]
        iterations = int(mdf["Iteration"].max())
        repeats = int(mdf["Repeat"].max())
        for rpt in range(1, repeats+1):
            rdf = mdf[mdf["Repeat"] == rpt]
            max_so_far = -np.inf
            y = []
            for it in range(1, iterations+1):
                cur = rdf[rdf["Iteration"] == it]["Fluorescence"].dropna()
                if cur.empty:
                    y.append(max_so_far if np.isfinite(max_so_far) else np.nan)
                    continue
                m = cur.max()
                if pd.notnull(m) and m > max_so_far:
                    max_so_far = m
                y.append(max_so_far)
            x = list(range(1, len(y)+1))
            plt.plot(x, y, label=f"{method} - Repeat {rpt}")
    plt.xlabel("Iteration")
    plt.ylabel("Highest Observed Fluorescence")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(n, iterations, repeat, methods, xlsx_path):
    df = _load_lnp_df(xlsx_path)
    results = {}
    if "Random" in methods:
        results.update(run_experiment(n, iterations, repeat, random_suggest, "Random", df))
    if "EDBO" in methods:
        delete_existing_optimization_files()
        results.update(run_experiment(n, iterations, repeat, edbo_suggest, "EDBO", df))
    if "BO" in methods:
        results.update(run_experiment(n, iterations, repeat, bo_suggest, "BO", df))
    if "LLM" in methods:
        results.update(run_experiment(n, iterations, repeat, llm_suggest, "LLM", df))
    if "R-LLM" in methods:
        results.update(run_experiment(n, iterations, repeat, reasoning_llm_suggest, "R-LLM", df))

    methods_str = "_".join(results.keys()) if results else "none"
    today = dt.date.today().strftime("%Y%m%d")
    filename = f"experiment_results_n{n}_iter{iterations}_rep{repeat}_{methods_str}_{today}.csv"
    saved = save_csv(results, filename, n, iterations, repeat)
    load_and_plot_summary(saved)
    load_and_plot_results(saved)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4, help="Batch size per iteration")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--methods", type=str, default="Random,BO,EDBO,LLM,R-LLM")
    p.add_argument("--xlsx", type=str, default="LNP.xlsx", help="Path to LNP.xlsx")
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    main(args.n, args.iterations, args.repeat, methods, args.xlsx)
