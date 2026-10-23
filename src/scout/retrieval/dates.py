from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from scout.retrieval.search import Hit, SearchBackend

_HOUR = r"(?:[01][0-9]|2[0-3])"
_CLOCK = rf"[T ]{_HOUR}:[0-5][0-9](?::[0-5][0-9](?:\.[0-9]+)?)?"
_TIME = rf"(?:{_CLOCK}(?:Z|[+-]{_HOUR}(?::?[0-5][0-9])?)?)?"
_PATTERN = re.compile(rf"([0-9]{{4}})(?:-?([0-9]{{2}})(?:-?([0-9]{{2}}){_TIME})?)?")
MAX_EXPANSION = 1000


@dataclass(frozen=True)
class Moment:
    first: date
    last: date
    text: str

    @property
    def exact(self) -> bool:
        return self.first == self.last


def parse_date(value: str) -> Moment | None:
    match = _PATTERN.fullmatch(value.strip()) if isinstance(value, str) else None
    if match is None:
        return None
    year, month, day = match.group(1), match.group(2), match.group(3)
    try:
        if month is None:
            return Moment(date(int(year), 1, 1), date(int(year), 12, 31), year)
        if day is None:
            last = calendar.monthrange(int(year), int(month))[1]
            return Moment(date(int(year), int(month), 1), date(int(year), int(month), last), f"{year}-{month}")
        exact = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return Moment(exact, exact, exact.isoformat())


def normalize_date(value: str) -> str:
    moment = parse_date(value)
    return moment.text if moment is not None else ""


def compare_dates(a: str, b: str) -> int:
    first, second = parse_date(a), parse_date(b)
    if first is None or second is None:
        return 0
    if first.first > second.last:
        return 1
    if first.last < second.first:
        return -1
    return 0


def newest(hits: Sequence[Hit]) -> Hit | None:
    best: Hit | None = None
    for hit in hits:
        if best is None or compare_dates(hit.passage.date, best.passage.date) > 0:
            best = hit
    return best


def render_today(today: str | date) -> str:
    return f"Today is {_full_day(today).isoformat()}."


def _full_day(today: str | date) -> date:
    if isinstance(today, date):
        return today
    moment = parse_date(today)
    if moment is None or not moment.exact:
        raise ValueError(f"{today!r} is not a full date like 2026-10-23")
    return moment.first


class AsOf:
    def __init__(
        self, backend: SearchBackend, today: str | date, keep_undated: bool = True, limit: int = MAX_EXPANSION
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.backend = backend
        self.today = _full_day(today)
        self.keep_undated = keep_undated
        self.limit = limit
        self.name = f"{backend.name}@{self.today.isoformat()}"

    def allowed(self, hit: Hit) -> bool:
        moment = parse_date(hit.passage.date)
        if moment is None:
            return self.keep_undated
        return moment.first <= self.today

    def search(self, query: str, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        fetch = k
        while True:
            hits = self.backend.search(query, fetch)
            kept = [hit for hit in hits if self.allowed(hit)]
            if len(kept) >= k or len(hits) < fetch or fetch >= self.limit:
                return kept[:k]
            fetch = min(fetch * 4, self.limit)
