# -*- coding: utf-8 -*-
"""
Читает books.csv и строит отдельные тепловые карты матриц сходства
по каждой из мер близости.
На осях: "№. Название (значение признака)".
Добавлена сортировка книг по осям и комбинированная мера.
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

# >>> ФЛАГ СОРТИРОВКИ <<<
SORT_BY_SIMILARITY = True          # True — сортировать книги по осям, False — исходный порядок

# Способ сортировки:
#   "mean"  — по среднему сходству с остальными (чем выше, тем «центральнее»)
#   "pca"   — по первой главной компоненте матрицы сходства (похожие рядом)
SORT_METHOD = "mean"

# ----------------------- 0b. Веса комбинированной меры -----------------------
# Сумма весов должна быть равна 1.0
W = {
    "tax":      0.35,   # таксономия
    "cycle":    0.05,   # часть цикла (покомпонентно)
    "trans":    0.05,   # перевод (покомпонентно)
    "year":     0.10,   # год публикации
    "pages":    0.05,   # количество страниц
    "age":      0.10,   # возрастное ограничение
    "title":    0.15,   # название
    "author":   0.15,   # автор
}
assert abs(sum(W.values()) - 1.0) < 1e-9, "Сумма весов должна быть 1.0"

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

# -------- Покомпонентные бинарные меры (для комбинированной) --------
def sim_cycle(i, j):
    return 1.0 if df.loc[i, "Часть цикла"] == df.loc[j, "Часть цикла"] else 0.0

def sim_trans(i, j):
    return 1.0 if df.loc[i, "Перевод"] == df.loc[j, "Перевод"] else 0.0

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

# ----------------------- 2b. Комбинированная мера -----------------------
def sim_combined(i, j):
    """
    Sim(d_i, d_j) = Σ_k w_k · sim_k(d_i, d_j)
    Веса w_k заданы в словаре W и нормированы (Σ w_k = 1).
    """
    return (
        W["tax"]    * sim_tax(i, j)      +
        W["cycle"]  * sim_cycle(i, j)    +
        W["trans"]  * sim_trans(i, j)    +
        W["year"]   * sim_year(df.loc[i, "Год публикации"], df.loc[j, "Год публикации"]) +
        W["pages"]  * sim_pages(df.loc[i, "Страницы"], df.loc[j, "Страницы"]) +
        W["age"]    * sim_age_lim(i, j)  +
        W["title"]  * sim_title(i, j)    +
        W["author"] * sim_author(i, j)
    )

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
    # >>> КОМБИНИРОВАННАЯ МЕРА <<<
        # >>> КОМБИНИРОВАННАЯ МЕРА <<<
    "combined": {
        "title": "Комбинированная мера (Σ wₖ · simₖ)",
        "matrix": build_matrix(sim_combined),
        "values": [""] * N,   # без вывода значений параметров
    },
}

# ----------------------- 4. Функция сортировки -----------------------
def get_order(M, method="mean"):
    """
    Возвращает список индексов, задающий порядок книг по осям.
    method="mean" — по среднему сходству с остальными (по убыванию).
    method="pca"  — по первой главной компоненте (по возрастанию, чтобы кластеры шли подряд).
    """
    if method == "mean":
        scores = M.mean(axis=1)
        return list(np.argsort(-scores))
    elif method == "pca":
        Mc = M - M.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        pc1 = U[:, 0] * S[0]
        return list(np.argsort(pc1))
    else:
        raise ValueError(f"Неизвестный метод сортировки: {method}")

# ----------------------- 5. Тепловые карты -----------------------
sns.set_theme(style="white")

for key, info in matrices.items():
    M = info["matrix"]
    values = info["values"]

    # --- сортировка осей ---
    if SORT_BY_SIMILARITY:
        order = get_order(M, method=SORT_METHOD)
        M_plot = M[np.ix_(order, order)]
        values_plot = [values[i] for i in order]
        idx_plot = [i for i in order]
    else:
        M_plot = M
        values_plot = values
        idx_plot = list(range(N))

    # Подписи: "№. Название (значение признака)"
    labels = [
        f"{idx_plot[k]+1}. {df.loc[idx_plot[k], 'Название']} ({values_plot[k]})"
        for k in range(N)
    ]

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        M_plot,
        ax=ax,
        cmap="viridis",
        vmin=0, vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Сходство"},
        square=True,
    )
    suffix = f" [sorted by {SORT_METHOD}]" if SORT_BY_SIMILARITY else ""
    ax.set_title(info["title"] + suffix, fontsize=20, pad=18)
    ax.tick_params(axis="x", rotation=90, labelsize=12)
    ax.tick_params(axis="y", rotation=0,  labelsize=12)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"heatmap_{key}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")

print("\nГотово. Все тепловые карты в папке:", OUT_DIR)