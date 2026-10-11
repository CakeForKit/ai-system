"""FastAPI-интерфейс рекомендательной системы книг."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
import secrets
import time
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.recommender import ContentRecommender, prepare_books, to_public_books


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
BOOKS_PATH = ROOT_DIR / "books.csv"
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("book_recommender")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

books = prepare_books(str(BOOKS_PATH))
recommender = ContentRecommender(books)
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Application started; books=%s", len(books))
    yield
    logger.info("Application stopped")


app = FastAPI(title="Книжный рекомендатель PCv3", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("APP_SESSION_SECRET", "development-" + secrets.token_urlsafe(24)))
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def user_state(request: Request) -> dict[str, dict[int, float]]:
    session_id = request.session.setdefault("session_id", secrets.token_urlsafe(18))
    states = getattr(app.state, "users", None)
    if states is None:
        app.state.users = {}
        states = app.state.users
    return states.setdefault(session_id, {"likes": {}, "dislikes": {}})


def selected_options(column: str) -> list[str]:
    return sorted(str(value) for value in books[column].dropna().unique())


FILTER_OPTIONS = {
    "sections": selected_options("Раздел"),
    "subsections": selected_options("Подраздел"),
    "cycle": ["да", "нет"],
    "translation": ["да", "нет"],
    "ages": sorted(int(value) for value in books["_age"].unique()),
}


def parse_number(value: str, label: str, minimum: int, maximum: int, errors: list[str]) -> int | None:
    if not value.strip():
        return None
    try:
        number = int(value)
    except ValueError:
        errors.append(f"«{label}» должно быть целым числом.")
        return None
    if not minimum <= number <= maximum:
        errors.append(f"«{label}» должно быть в диапазоне от {minimum} до {maximum}.")
    return number


def search_books(filters: dict[str, str]) -> tuple[list[int], list[str]]:
    errors: list[str] = []
    query = filters["query"].strip()
    if len(query) > 100:
        errors.append("Поисковая строка не должна быть длиннее 100 символов.")

    year_from = parse_number(filters["year_from"], "Год от", int(books["Год публикации"].min()), int(books["Год публикации"].max()), errors)
    year_to = parse_number(filters["year_to"], "Год до", int(books["Год публикации"].min()), int(books["Год публикации"].max()), errors)
    pages_from = parse_number(filters["pages_from"], "Страниц от", 1, int(books["_pages"].max()), errors)
    pages_to = parse_number(filters["pages_to"], "Страниц до", 1, int(books["_pages"].max()), errors)
    if year_from is not None and year_to is not None and year_from > year_to:
        errors.append("Нижняя граница года не может быть больше верхней.")
    if pages_from is not None and pages_to is not None and pages_from > pages_to:
        errors.append("Нижняя граница количества страниц не может быть больше верхней.")
    for field, options in (("section", FILTER_OPTIONS["sections"]), ("subsection", FILTER_OPTIONS["subsections"]), ("cycle", FILTER_OPTIONS["cycle"]), ("translation", FILTER_OPTIONS["translation"])):
        if filters[field] and filters[field] not in options:
            errors.append(f"Недопустимое значение фильтра «{field}».")

    if errors:
        return [], errors

    result = books
    if query:
        needle = query.casefold()
        result = result[result["Название"].str.casefold().str.contains(needle, regex=False) | result["Автор"].str.casefold().str.contains(needle, regex=False)]
    if year_from is not None:
        result = result[result["Год публикации"] >= year_from]
    if year_to is not None:
        result = result[result["Год публикации"] <= year_to]
    if pages_from is not None:
        result = result[result["_pages"] >= pages_from]
    if pages_to is not None:
        result = result[result["_pages"] <= pages_to]
    if filters["age"]:
        result = result[result["_age"] == int(filters["age"])]
    if filters["section"]:
        result = result[result["Раздел"] == filters["section"]]
    if filters["subsection"]:
        result = result[result["Подраздел"] == filters["subsection"]]
    if filters["cycle"]:
        result = result[result["Часть цикла"] == filters["cycle"]]
    if filters["translation"]:
        result = result[result["Перевод"] == filters["translation"]]
    return result["id"].astype(int).tolist(), []


def render_page(request: Request, book_ids: list[int], *, title: str, view: str, state: dict, errors: list[str] | None = None, filters: dict[str, str] | None = None, scores: dict[int, float] | None = None) -> HTMLResponse:
    book_list = to_public_books(books, book_ids)
    for book in book_list:
        book["reaction"] = "like" if book["id"] in state["likes"] else "dislike" if book["id"] in state["dislikes"] else None
        book["score"] = scores.get(book["id"]) if scores else None
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": title,
            "books": book_list,
            "view": view,
            "errors": errors or [],
            "filters": filters or {},
            "options": FILTER_OPTIONS,
            "likes_count": len(state["likes"]),
            "dislikes_count": len(state["dislikes"]),
        },
    )


@app.get("/", response_class=HTMLResponse)
async def catalog(request: Request) -> HTMLResponse:
    state = user_state(request)
    return render_page(request, books["id"].astype(int).tolist(), title="Все книги", view="catalog", state=state)


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: Annotated[str, Form()] = "",
    year_from: Annotated[str, Form()] = "",
    year_to: Annotated[str, Form()] = "",
    pages_from: Annotated[str, Form()] = "",
    pages_to: Annotated[str, Form()] = "",
    age: Annotated[str, Form()] = "",
    section: Annotated[str, Form()] = "",
    subsection: Annotated[str, Form()] = "",
    cycle: Annotated[str, Form()] = "",
    translation: Annotated[str, Form()] = "",
    scope: Annotated[str, Form()] = "catalog",
) -> HTMLResponse:
    filters = locals().copy()
    filters = {key: filters[key] for key in ("query", "year_from", "year_to", "pages_from", "pages_to", "age", "section", "subsection", "cycle", "translation")}
    state = user_state(request)
    book_ids, errors = search_books(filters)
    if scope not in {"catalog", "recommendations"}:
        errors.append("Недопустимая область поиска.")
    if scope == "recommendations" and not errors:
        ranked = recommender.rank(state["likes"], state["dislikes"], time.time())
        allowed_ids = set(book_ids)
        book_ids = [item.book_id for item in ranked if item.book_id in allowed_ids]
        scores = {item.book_id: item.score for item in ranked}
        logger.info("Recommendation search: filters=%s; results=%s", filters, book_ids)
        return render_page(
            request,
            book_ids,
            title="Лента рекомендаций PCv3 — результаты поиска",
            view="recommendations",
            state=state,
            errors=errors,
            filters=filters,
            scores=scores,
        )
    if not errors:
        logger.info("Catalog search: filters=%s; results=%s", filters, book_ids)
    return render_page(
        request,
        book_ids,
        title="Каталог — результаты поиска",
        view="catalog",
        state=state,
        errors=errors,
        filters=filters,
    )


@app.post("/feedback/{book_id}")
async def feedback(request: Request, book_id: int, action: Annotated[str, Form()], return_to: Annotated[str, Form()] = "/") -> RedirectResponse:
    if book_id not in set(books["id"]):
        return RedirectResponse("/", status_code=303)
    if action not in {"like", "dislike"}:
        return RedirectResponse(return_to, status_code=303)
    state = user_state(request)
    state["likes"].pop(book_id, None)
    state["dislikes"].pop(book_id, None)
    state[f"{action}s"][book_id] = time.time()
    logger.info("Feedback: book_id=%s; action=%s", book_id, action)
    return RedirectResponse(return_to, status_code=303)


@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations(request: Request) -> HTMLResponse:
    state = user_state(request)
    ranked = recommender.rank(state["likes"], state["dislikes"], time.time())
    ids = [item.book_id for item in ranked]
    scores = {item.book_id: item.score for item in ranked}
    logger.info("Recommendations: likes=%s; dislikes=%s; results=%s", sorted(state["likes"]), sorted(state["dislikes"]), [(item.book_id, round(item.score, 4)) for item in ranked])
    return render_page(request, ids, title="Лента рекомендаций PCv3", view="recommendations", state=state, scores=scores)
