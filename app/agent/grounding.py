"""grounding 응답 파싱 — 0~1000 정규화 좌표 → 이미지 대비 분수(0~1) 박스.

참고 구현(vectra-web `parseVisualInspection`)과 같은 규칙:
  - 코드펜스를 벗기고 JSON으로 읽는다. 실패하면 본문에서 JSON 객체를 찾아 다시 읽는다.
  - bbox는 [x1, y1, x2, y2]. 0~1000으로 자르고, 뒤집혔거나 넓이가 0인 박스는 버린다.
  - 결과는 {x, y, w, h}(0~1 분수)라서 프론트는 그대로 %로 바꿔 오버레이하면 된다.
추가로, 지시를 어기고 자기 학습 포맷으로 답하는 VLM을 위해 몇 가지 흔한 변형을 받아 준다(아래 주석).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .. import config
from .prompts import REGION_TYPES

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass
class VisualInspection:
    text: str = ""
    boxes: list[dict[str, Any]] = field(default_factory=list)
    structured: bool = False


def _load_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", raw):
        try:
            value, _ = decoder.raw_decode(raw, match.start())
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def _coordinates(region: dict[str, Any]) -> list[float] | None:
    """[x1, y1, x2, y2] 순서의 숫자 4개를 돌려준다.

    - "bbox"    : 우리가 요청한 형식 (x1, y1, x2, y2)
    - "bbox_2d" : Qwen-VL 계열이 즐겨 쓰는 키 (x1, y1, x2, y2)
    - "box_2d"  : Gemini 고유 형식 (y1, x1, y2, x2) → 순서를 바꿔 준다
    """
    for key, y_first in (("bbox", False), ("bbox_2d", False), ("box_2d", True)):
        raw = region.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            try:
                values = [float(item) for item in raw]
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(item) for item in values):
                return None
            return [values[1], values[0], values[3], values[2]] if y_first else values
    return None


def _to_thousandths(values: list[float], width: int | None, height: int | None) -> list[float]:
    """좌표 단위를 0~1000으로 맞춘다.

    - 모두 1 이하         → 0~1 분수로 답한 것 → ×1000
    - 1000 초과 값이 있음  → 픽셀 좌표로 답한 것 → 보낸 이미지 크기로 나눈다(크기를 모르면 그대로 두고 아래에서 자른다)
    """
    if max(values) <= 1.0:
        return [item * 1000.0 for item in values]
    if max(values) > 1000.0 and width and height:
        return [values[0] / width * 1000.0, values[1] / height * 1000.0,
                values[2] / width * 1000.0, values[3] / height * 1000.0]
    return values


def _region_type(value: Any) -> str:
    """허용된 타입 중 하나로 정규화한다. "text|object|table"처럼 스키마를 베낀 값은 첫 유효 토큰을 쓴다."""
    for token in re.split(r"[|/,\s]+", str(value or "").lower()):
        if token in REGION_TYPES:
            return token
    return "other"


def parse_visual_inspection(value: Any, *, image_width: int | None = None,
                            image_height: int | None = None) -> VisualInspection:
    original = _THINK_BLOCK.sub("", str(value or "")).strip()
    raw = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", original))
    parsed = _load_json(raw)
    if parsed is None:
        return VisualInspection(text=original, boxes=[], structured=False)
    if isinstance(parsed, list):
        parsed = {"text": "", "regions": parsed}
    regions = parsed.get("regions")
    boxes: list[dict[str, Any]] = []
    for region in (regions if isinstance(regions, list) else [])[: config.MAX_GROUNDING_REGIONS]:
        if not isinstance(region, dict):
            continue
        values = _coordinates(region)
        if values is None:
            continue
        x1, y1, x2, y2 = (max(0.0, min(1000.0, item)) for item in _to_thousandths(values, image_width, image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        box: dict[str, Any] = {
            "x": x1 / 1000.0, "y": y1 / 1000.0, "w": (x2 - x1) / 1000.0, "h": (y2 - y1) / 1000.0,
            "label": str(region.get("label") or "")[:80],
            "type": _region_type(region.get("type")),
        }
        try:
            confidence = float(region.get("confidence"))
            if math.isfinite(confidence):
                box["confidence"] = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            pass
        boxes.append(box)
    return VisualInspection(text=str(parsed.get("text") or ""), boxes=boxes, structured=True)


def valid_box(box: dict[str, Any]) -> bool:
    epsilon = 1e-6
    return (box["w"] > 0 and box["h"] > 0 and box["x"] >= 0 and box["y"] >= 0
            and box["x"] + box["w"] <= 1 + epsilon and box["y"] + box["h"] <= 1 + epsilon)


def map_box_to_source(box: dict[str, Any], source_box: tuple[float, float, float, float]) -> dict[str, Any]:
    """타일(부분 이미지) 기준 박스를 원본 전체 기준으로 옮긴다. 전체 이미지(0,0,1,1)면 그대로다."""
    x0, y0, x1, y1 = source_box
    width, height = x1 - x0, y1 - y0
    return {**box, "x": x0 + box["x"] * width, "y": y0 + box["y"] * height,
            "w": box["w"] * width, "h": box["h"] * height}
