"""Pydantic validation helpers used at service and agent boundaries."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from backend.exceptions import DomainValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def validate_model(model_cls: Type[T], data: Any, *, context: str = "") -> T:
    """Validate *data* with *model_cls* or raise ``DomainValidationError``."""
    try:
        return model_cls.model_validate(data)
    except PydanticValidationError as exc:
        prefix = f"{context}: " if context else ""
        raise DomainValidationError(f"{prefix}{exc}") from exc


def try_validate_model(model_cls: Type[T], data: Any, *, context: str = "") -> Optional[T]:
    """Validate *data* or return ``None`` and log when invalid."""
    try:
        return validate_model(model_cls, data, context=context)
    except DomainValidationError as exc:
        logger.warning("%s", exc.message)
        return None


def parse_models(
    model_cls: Type[T],
    items: List[Any],
    *,
    context: str = "",
    on_item: Callable[[Any], Any] | None = None,
) -> Tuple[List[T], List[str]]:
    """Parse a list of dicts into models, skipping invalid items."""
    parsed: List[T] = []
    skipped: List[str] = []
    for item in items:
        payload = on_item(item) if on_item else item
        try:
            parsed.append(model_cls.model_validate(payload))
        except PydanticValidationError:
            name = "?"
            if isinstance(item, dict):
                name = str(item.get("model_name") or item.get("model") or item.get("name", "?"))
            skipped.append(name)
            logger.warning("%s: skipped invalid item %r", context or model_cls.__name__, name)
    return parsed, skipped


def parse_tool_results(
    model_cls: Type[T],
    items: List[Any],
    *,
    context: str = "",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse tool outputs into validated dicts, skipping invalid items."""
    parsed, skipped = parse_models(model_cls, items, context=context)
    return [item.model_dump() for item in parsed], skipped
