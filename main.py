# -*- coding: utf-8 -*-
"""
Читает books.csv и строит отдельные тепловые карты матриц сходства
по каждой из мер близости.
На осях: "№. Название (значение признака)".
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log, exp
from Levenshtein import distance as lev_distance  # pip install python-Levenshtein

# ----------------------- 0. Настройки -----------------------
CSV_PATH = "books.csv"
OUT_DIR = "heatmaps"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------- 1. Загрузка -----------------------
df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
N = len(df)

df["Год публикации"] = df["Год публикации"].astype(int)
df["Страницы"] = df["Кол-во страниц (объём)"].astype(str).str.replace("~", "").astype(int)
df["Возраст"] = df["Возрастное ограничение"].astype(str).str.replace("+", "").astype(int)
df["Часть цикла"] = (df["Часть цикла"] == "да").astype(int)
df["Перевод"] = (df["Перевод"] == "да").astype(int)

# ----------------------- 2. Меры близости -----------------------
Y_NOW = 2026
Y_MIN = df["Год публикации"].min()

def sim_year(yi, yj):
    dist = abs(log(Y_NOW - yi + 1) - log(Y_NOW - yj + 1))
    dist_max = log(Y_NOW - Y_MIN + 1)
    return 1.0 - dist / dist_max

def sim_pages(pi, pj, tau=1.0):
    return exp(-abs(log(pi) - log(pj)) / tau)

def sim_age_limit(ai, aj, tau=6.0):
    return exp(-abs(ai - aj) / tau)

# -------- Таксономия --------
def cat_path(row):
    parts = ["Таксономия", row["Раздел"]]
    if row["Подраздел"] != "—":
        parts.append(row["Подраздел"])
    if row["Подподраздел"] != "—":
        parts.append(row["Подподраздел"])
    return parts

paths = [cat_path(r) for _, r in df.iterrows()]

def depth(path):
    return len(path) - 1

def lca_depth(p1, p2):
    d = 0
    for a, b in zip(p1, p2):
        if a == b:
            d += 1
        else:
            break
    return d - 1

def sim_tax(i, j):
    p1, p2 = paths[i], paths[j]
    d_lca = lca_depth(p1, p2)
    d_max = max(depth(p1), depth(p2))
    return 1.0 if d_max == 0 else max(0.0, d_lca / d_max)

def sim_bin(i, j):
    b1 = np.array([df.loc[i, "Часть цикла"], df.loc[i, "Перевод"]])
    b2 = np.array([df.loc[j, "Часть цикла"], df.loc[j, "Перевод"]])
    return float(np.mean(b1 == b2))

def sim_age_lim(i, j):
    return sim_age_limit(df.loc[i, "Возраст"], df.loc[j, "Возраст"], tau=6.0)

def authors_set(s):
    return set(a.strip() for a in str(s).replace(",", ";").split(";") if a.strip())

def sim_author(i, j):
    A = authors_set(df.loc[i, "Автор"])
    B = authors_set(df.loc[j, "Автор"])
    return len(A & B) / len(A | B) if (A | B) else 0.0

def sim_title(i, j):
    t1, t2 = str(df.loc[i, "Название"]), str(df.loc[j, "Название"])
    if not t1 and not t2:
        return 1.0
    return 1.0 - lev_distance(t1, t2) / max(len(t1), len(t2))

# ----------------------- 3. Построение матриц -----------------------
def build_matrix(fn):
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = fn(i, j)
    return M

matrices = {
    "year": {
        "title": "Год публикации (лог. возраст)",
        "matrix": build_matrix(lambda i, j: sim_year(df.loc[i, "Год публикации"], df.loc[j, "Год публикации"])),
        "values": df["Год публикации"].tolist(),
    },
    "pages": {
        "title": "Кол-во страниц (лог. объём)",
        "matrix": build_matrix(lambda i, j: sim_pages(df.loc[i, "Страницы"], df.loc[j, "Страницы"])),
        "values": df["Страницы"].tolist(),
    },
    "age_limit": {
        "title": "Возрастное ограничение",
        "matrix": build_matrix(sim_age_lim),
        "values": df["Возрастное ограничение"].tolist(),
    },
    "taxonomy": {
        "title": "Таксономия (LCA)",
        "matrix": build_matrix(sim_tax),
        "values": [
            " / ".join(p[1:]) if len(p) > 1 else "—" for p in paths
        ],
    },
    "binary": {
        "title": "Бинарные признаки",
        "matrix": build_matrix(sim_bin),
        "values": [
            f"цикл={df.loc[i, 'Часть цикла']}, пер={df.loc[i, 'Перевод']}"
            for i in range(N)
        ],
    },
    "author": {
        "title": "Автор (Жаккар)",
        "matrix": build_matrix(sim_author),
        "values": df["Автор"].tolist(),
    },
    "title_sim": {
        "title": "Название (Левенштейн)",
        "matrix": build_matrix(sim_title),
        "values": df["Название"].tolist(),
    },
}

# ----------------------- 4. Тепловые карты (по одной на файл) -----------------------
sns.set_theme(style="white")

for key, info in matrices.items():
    # Подписи: "№. Название (значение признака)"
    labels = [
        f"{i+1}. {df.loc[i, 'Название']} ({info['values'][i]})"
        for i in range(N)
    ]

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        info["matrix"],
        ax=ax,
        cmap="viridis",
        vmin=0, vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Сходство"},
        square=True,
    )
    ax.set_title(info["title"], fontsize=20, pad=18)
    ax.tick_params(axis="x", rotation=90, labelsize=12)
    ax.tick_params(axis="y", rotation=0,  labelsize=12)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"heatmap_{key}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")

print("\nГотово. Все тепловые карты в папке:", OUT_DIR)