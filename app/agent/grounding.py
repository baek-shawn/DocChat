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
    # 출력 상한에 닿아 끊긴 호출(Step 6-0). 끊긴 글은 추론이거나 미완성 JSON이라 결과로 쓰지 않는다 → text·boxes는 비어 있다.
    cut_off: bool = False
    # 추론이 끝나지 않아(예산 초과·반복) 이어 쓰기로도 답을 받지 못한 호출(Step 6). 역시 다시 보내지 않는다.
    runaway: bool = False


def _load_json(raw: str) -> Any:
    """객체나 배열만 돌려준다. 그 밖의 JSON 값은 구조화된 답이 아니다.

    `3600`, `"none"`, `true`처럼 값 하나만 온 응답도 JSON으로는 읽힌다 — 그대로 돌려주면 뒤에서 객체로 다루다 예외가 난다.
    """
    try:
        value = json.loads(raw)
        if isinstance(value, (dict, list)):
            return value
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


# --------------------------------------------------------------------------- 타일 박스 병합
def _overlap(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float]:
    """(IoU, 작은 박스가 겹친 비율)"""
    width = min(first["x"] + first["w"], second["x"] + second["w"]) - max(first["x"], second["x"])
    height = min(first["y"] + first["h"], second["y"] + second["h"]) - max(first["y"], second["y"])
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    shared = width * height
    first_area, second_area = first["w"] * first["h"], second["w"] * second["h"]
    smaller = min(first_area, second_area)
    union = first_area + second_area - shared
    return (shared / union if union > 0 else 0.0), (shared / smaller if smaller > 0 else 0.0)


def _rank(box: dict[str, Any]) -> tuple[float, float]:
    confidence = box.get("confidence")
    return (float(confidence) if isinstance(confidence, (int, float)) else -1.0, box["w"] * box["h"])


def _combine(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """영역은 두 박스를 모두 덮는 사각형, 라벨·종류·신뢰도는 더 믿을 만한 쪽(신뢰도 → 넓이 순)의 것."""
    x0, y0 = min(first["x"], second["x"]), min(first["y"], second["y"])
    x1 = max(first["x"] + first["w"], second["x"] + second["w"])
    y1 = max(first["y"] + first["h"], second["y"] + second["h"])
    better = first if _rank(first) >= _rank(second) else second
    return {**better, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def merge_tile_boxes(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """타일별 박스 목록(이미 전체 좌표로 옮긴 것)을 한 목록으로 합친다.

    타일은 서로 겹치므로 겹침 영역에 있는 대상은 두 타일에서 한 번씩, 모두 두 번 잡힌다. **서로 다른 타일**에서 나온
    **같은 종류**의 박스가 아래 중 하나를 만족하면 같은 대상으로 보고 하나로 합친다.
      - IoU ≥ TILE_BOX_MERGE_IOU                         (두 타일이 대상을 온전히 봤다)
      - 작은 박스의 TILE_BOX_MERGE_CONTAINMENT 이상이 겹침  (한 타일은 가장자리에서 잘린 일부만 봤다)
    같은 타일 안의 박스끼리는 합치지 않는다 — 모델이 한 이미지에서 따로 잡은 것은 전체 모드와 똑같이 그대로 둔다.
    """
    merged: list[tuple[set[int], dict[str, Any]]] = []
    for index, boxes in enumerate(groups):
        for box in boxes:
            match = None
            for position, (owners, existing) in enumerate(merged):
                if index in owners or existing.get("type") != box.get("type"):
                    continue
                iou, contained = _overlap(existing, box)
                if iou >= config.TILE_BOX_MERGE_IOU or contained >= config.TILE_BOX_MERGE_CONTAINMENT:
                    match = position
                    break
            if match is None:
                merged.append(({index}, dict(box)))
            else:
                owners, existing = merged[match]
                merged[match] = (owners | {index}, _combine(existing, box))
    return [box for _owners, box in merged]
