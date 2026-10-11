"""Контентная мера сходства и ранжирование для FastAPI-приложения."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, log
import re
from typing import Iterable, Mapping

import pandas as pd
import torch
import torch.nn.functional as functional


def tokenize_title(title: object) -> list[str]:
    normalized = str(title).lower().replace("ё", "е")
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def authors_set(authors: object) -> set[str]:
    return {
        author.strip().lower()
        for author in str(authors).replace(",", ";").split(";")
        if author.strip()
    }


@dataclass(frozen=True)
class RankedBook:
    book_id: int
    score: float


class ContentRecommender:
    """Вычисляет комбинированную близость книг из books.csv."""

    def __init__(
        self,
        books: pd.DataFrame,
        similarity_weights: Mapping[str, float],
        like_max_similarity_weight: float,
        dislike_penalty: float,
        time_decay_tau_seconds: float,
        time_decay_min_weight: float,
    ) -> None:
        self.books = books.reset_index(drop=True).copy()
        self.weights = dict(similarity_weights)
        self.like_max_similarity_weight = like_max_similarity_weight
        self.dislike_penalty = dislike_penalty
        self.time_decay_tau_seconds = time_decay_tau_seconds
        self.time_decay_min_weight = time_decay_min_weight
        self.size = len(self.books)
        self.now_year = datetime.now().year
        self.min_year = int(self.books["Год публикации"].min())
        self.paths = [self._category_path(row) for _, row in self.books.iterrows()]
        self.title_tokens = [tokenize_title(title) for title in self.books["Название"]]
        self.title_similarity = self._build_title_similarity()

    @staticmethod
    def _category_path(row: pd.Series) -> list[str]:
        path = ["Таксономия", row["Раздел"]]
        for field in ("Подраздел", "Подподраздел"):
            if row[field] != "—":
                path.append(row[field])
        return path

    def _build_title_similarity(self) -> torch.Tensor:
        vocabulary = sorted({token for tokens in self.title_tokens for token in tokens})
        token_index = {token: index for index, token in enumerate(vocabulary)}
        tf = torch.zeros((self.size, len(vocabulary)), dtype=torch.float32)
        for row_index, tokens in enumerate(self.title_tokens):
            for token in tokens:
                tf[row_index, token_index[token]] += 1

        document_frequency = (tf > 0).sum(dim=0)
        idf = torch.log((self.size + 1) / (document_frequency + 1)) + 1
        vectors = functional.normalize(tf * idf, p=2, dim=1)
        return torch.clamp(vectors @ vectors.T, 0.0, 1.0)

    def _lca_depth(self, first: list[str], second: list[str]) -> int:
        common = 0
        for left, right in zip(first, second):
            if left != right:
                break
            common += 1
        return common - 1

    def _sim_taxonomy(self, first: int, second: int) -> float:
        path_a, path_b = self.paths[first], self.paths[second]
        max_depth = max(len(path_a) - 1, len(path_b) - 1)
        return 1.0 if max_depth == 0 else max(0.0, self._lca_depth(path_a, path_b) / max_depth)

    def _sim_year(self, first: int, second: int) -> float:
        year_a = int(self.books.at[first, "Год публикации"])
        year_b = int(self.books.at[second, "Год публикации"])
        distance = abs(log(self.now_year - year_a + 1) - log(self.now_year - year_b + 1))
        maximum = log(self.now_year - self.min_year + 1)
        return 1.0 - distance / maximum if maximum else 1.0

    def _sim_pages(self, first: int, second: int) -> float:
        pages_a = int(self.books.at[first, "_pages"])
        pages_b = int(self.books.at[second, "_pages"])
        return exp(-abs(log(pages_a) - log(pages_b)))

    def _sim_age(self, first: int, second: int) -> float:
        age_a = int(self.books.at[first, "_age"])
        age_b = int(self.books.at[second, "_age"])
        return exp(-abs(age_a - age_b) / 6.0)

    def _sim_author(self, first: int, second: int) -> float:
        authors_a = authors_set(self.books.at[first, "Автор"])
        authors_b = authors_set(self.books.at[second, "Автор"])
        return len(authors_a & authors_b) / len(authors_a | authors_b) if authors_a | authors_b else 0.0

    def similarity(self, first_id: int, second_id: int) -> float:
        """Возвращает Sim(d_i, d_j) из README для идентификаторов книг от 1."""
        first, second = first_id - 1, second_id - 1
        cycle = float(self.books.at[first, "_cycle"] == self.books.at[second, "_cycle"])
        translation = float(self.books.at[first, "_translation"] == self.books.at[second, "_translation"])
        title = 1.0 if not self.title_tokens[first] and not self.title_tokens[second] else float(self.title_similarity[first, second])
        return (
            self.weights["tax"] * self._sim_taxonomy(first, second)
            + self.weights["cycle"] * cycle
            + self.weights["translation"] * translation
            + self.weights["year"] * self._sim_year(first, second)
            + self.weights["pages"] * self._sim_pages(first, second)
            + self.weights["age"] * self._sim_age(first, second)
            + self.weights["title"] * title
            + self.weights["author"] * self._sim_author(first, second)
        )

    def _recency_weight(self, action_time: float, now: float, first_action_time: float) -> float:
        """Логарифмическое затухание из формулы PCv3 в README."""
        elapsed_since_first = max(0.0, now - first_action_time)
        if elapsed_since_first == 0:
            return 1.0
        denominator = log(1 + elapsed_since_first / self.time_decay_tau_seconds)
        elapsed_since_action = max(0.0, now - action_time)
        numerator = log(1 + elapsed_since_action / self.time_decay_tau_seconds)
        weight = self.time_decay_min_weight + (1 - self.time_decay_min_weight) * (1 - numerator / denominator)
        return min(1.0, max(self.time_decay_min_weight, weight))

    def relevance_breakdown(
        self,
        candidate_id: int,
        likes: dict[int, float],
        dislikes: dict[int, float],
        now: float,
    ) -> dict[str, object]:
        """Возвращает слагаемые PCv3-формулы для объяснения рекомендации."""
        all_action_times = [*likes.values(), *dislikes.values()]
        first_action_time = min(all_action_times) if all_action_times else now

        def contributions(reactions: dict[int, float]) -> list[dict[str, float | int]]:
            result = []
            for book_id, action_time in reactions.items():
                weight = self._recency_weight(action_time, now, first_action_time)
                similarity = self.similarity(book_id, candidate_id)
                result.append(
                    {
                        "book_id": book_id,
                        "weight": weight,
                        "similarity": similarity,
                        "contribution": weight * similarity,
                    }
                )
            return result

        like_items = contributions(likes)
        dislike_items = contributions(dislikes)
        positive_numerator = sum(float(item["contribution"]) for item in like_items)
        negative_numerator = sum(float(item["contribution"]) for item in dislike_items)
        positive_denominator = sum(float(item["weight"]) for item in like_items)
        negative_denominator = sum(float(item["weight"]) for item in dislike_items)
        weighted_average = positive_numerator / positive_denominator if positive_denominator else 0.0
        maximum_similarity = max((float(item["similarity"]) for item in like_items), default=0.0)
        positive = (
            self.like_max_similarity_weight * maximum_similarity
            + (1 - self.like_max_similarity_weight) * weighted_average
        )
        negative = negative_numerator / negative_denominator if negative_denominator else 0.0
        return {
            "likes": like_items,
            "dislikes": dislike_items,
            "positive_numerator": positive_numerator,
            "positive_denominator": positive_denominator,
            "like_max_similarity": maximum_similarity,
            "like_weighted_average": weighted_average,
            "positive": positive,
            "negative_numerator": negative_numerator,
            "negative_denominator": negative_denominator,
            "negative": negative,
            "alpha": self.like_max_similarity_weight,
            "penalty": self.dislike_penalty,
            "score": positive - self.dislike_penalty * negative,
        }

    def rank(
        self,
        likes: dict[int, float],
        dislikes: dict[int, float],
        now: float,
        limit: int | None = None,
    ) -> list[RankedBook]:
        """Ранжирует непрореагированные книги по PCv3-оценке релевантности."""
        if not likes:
            return []

        excluded = set(likes) | set(dislikes)
        ranked: list[RankedBook] = []

        for candidate_id in range(1, self.size + 1):
            if candidate_id in excluded:
                continue
            breakdown = self.relevance_breakdown(candidate_id, likes, dislikes, now)
            ranked.append(RankedBook(candidate_id, float(breakdown["score"])))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked if limit is None else ranked[:limit]


def prepare_books(csv_path: str) -> pd.DataFrame:
    books = pd.read_csv(csv_path, sep=";", encoding="utf-8").copy()
    books.insert(0, "id", range(1, len(books) + 1))
    books["_pages"] = books["Кол-во страниц (объём)"].astype(str).str.replace("~", "", regex=False).astype(int)
    books["_age"] = books["Возрастное ограничение"].astype(str).str.replace("+", "", regex=False).astype(int)
    books["_cycle"] = books["Часть цикла"].eq("да")
    books["_translation"] = books["Перевод"].eq("да")
    return books


def to_public_books(books: pd.DataFrame, book_ids: Iterable[int]) -> list[dict[str, object]]:
    indexed = books.set_index("id", drop=False)
    result: list[dict[str, object]] = []
    for book_id in book_ids:
        row = indexed.loc[book_id]
        result.append(
            {
                "id": int(row["id"]),
                "title": row["Название"],
                "author": row["Автор"],
                "section": row["Раздел"],
                "subsection": row["Подраздел"],
                "year": int(row["Год публикации"]),
                "pages": int(row["_pages"]),
                "age": int(row["_age"]),
                "cycle": row["Часть цикла"],
                "translation": row["Перевод"],
            }
        )
    return result
