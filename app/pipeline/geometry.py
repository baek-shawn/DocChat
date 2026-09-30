"""크기 계산만 하는 순수 함수 — 이미지 한도(§5.2)와 타일 분할(Step 5).

PDF 렌더(`pdf.py`)와 이미지 처리(`images.py`)가 같은 계산을 쓰도록 여기로 모았다. 라이브러리에 의존하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import config


def cap_scale(width: float, height: float) -> float:
    """비율을 유지한 채 긴 변 ≤ MAX_VISION_IMAGE_EDGE, 픽셀 수 ≤ MAX_VISION_IMAGE_PIXELS가 되는 배율(≤ 1)."""
    if width <= 0 or height <= 0:
        return 1.0
    edge_scale = config.MAX_VISION_IMAGE_EDGE / max(width, height)
    pixel_scale = math.sqrt(config.MAX_VISION_IMAGE_PIXELS / (width * height))
    return min(1.0, edge_scale, pixel_scale)


def capped_size(width: int, height: int) -> tuple[int, int, float]:
    scale = cap_scale(width, height)
    if scale >= 1.0:
        return width, height, 1.0
    # 5000 * (3072/5000) = 3071.9999…처럼 부동소수 오차로 1px를 잃지 않도록 아주 작은 여유를 준다.
    new_width = max(1, int(width * scale + 1e-6))
    new_height = max(1, int(height * scale + 1e-6))
    while new_width * new_height > config.MAX_VISION_IMAGE_PIXELS or max(new_width, new_height) > config.MAX_VISION_IMAGE_EDGE:
        if new_width >= new_height:
            new_width -= 1
        else:
            new_height -= 1
    return new_width, new_height, scale


# --------------------------------------------------------------------------- 타일 분할
@dataclass(frozen=True)
class TileBox:
    """타일 하나. 좌표는 분할 기준 해상도(TilePlan.width × height)의 픽셀이고, row·col은 1부터 센다."""
    row: int
    col: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class TilePlan:
    width: int            # 분할 기준 해상도. 타일 수 상한 때문에 원본보다 작아질 수 있다(scale < 1)
    height: int
    scale: float          # 원본 크기 대비 배율
    rows: int
    cols: int
    tiles: tuple[TileBox, ...]      # 읽기 순서: 위 → 아래, 왼쪽 → 오른쪽

    def source_box(self, tile: TileBox) -> tuple[float, float, float, float]:
        """타일이 원본에서 차지하는 영역(정규화 x0, y0, x1, y1)."""
        return (tile.left / self.width, tile.top / self.height, tile.right / self.width, tile.bottom / self.height)


def _split_axis(length: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """한 축을 같은 크기의 구간으로 고르게 나눈다. 구간은 tile_size 이하이고 이웃과 overlap 이상 겹친다."""
    if length <= tile_size:
        return [(0, length)]
    count = math.ceil((length - overlap) / (tile_size - overlap))
    size = math.ceil((length + (count - 1) * overlap) / count)
    step = (length - size) / (count - 1)
    starts = [round(index * step) for index in range(count)]
    return [(start, start + size) for start in starts]


def needs_tiling(width: float, height: float) -> bool:
    """원본이 타일 기준 크기보다 작으면 나누지 않는다 — 전체 모드와 똑같이 처리한다."""
    return max(width, height) > config.TILE_MIN_SOURCE_EDGE and max(width, height) > config.TILE_SIZE


def plan_tiles(width: int, height: int, *, tile_size: int | None = None, overlap: float | None = None,
               max_tiles: int | None = None) -> TilePlan:
    """width × height 영역을 겹치는 타일로 나눈다.

    타일 수가 max_tiles를 넘으면 기준 해상도를 조금씩 낮춰 맞춘다(호출 수가 끝없이 늘지 않게).
    """
    tile_size = tile_size or config.TILE_SIZE
    ratio = config.TILE_OVERLAP if overlap is None else overlap
    max_tiles = max_tiles or config.MAX_TILES_PER_IMAGE
    # 겹침이 타일의 절반을 넘으면 같은 자리를 세 번 이상 보게 된다.
    overlap_px = min(int(round(tile_size * ratio)), tile_size // 2)
    width, height = max(1, int(width)), max(1, int(height))

    scale = 1.0
    while True:
        scaled_width, scaled_height = max(1, round(width * scale)), max(1, round(height * scale))
        columns = _split_axis(scaled_width, tile_size, overlap_px)
        rows = _split_axis(scaled_height, tile_size, overlap_px)
        if len(rows) * len(columns) <= max_tiles or max(scaled_width, scaled_height) <= tile_size:
            break
        scale *= 0.95
    tiles = tuple(
        TileBox(row=row_index + 1, col=col_index + 1, left=left, top=top, right=right, bottom=bottom)
        for row_index, (top, bottom) in enumerate(rows)
        for col_index, (left, right) in enumerate(columns)
    )
    return TilePlan(width=scaled_width, height=scaled_height, scale=scale, rows=len(rows), cols=len(columns), tiles=tiles)


# --------------------------------------------------------------------------- 타일 결과
@dataclass
class RenderedTile:
    row: int
    col: int
    box: tuple[float, float, float, float]    # 원본 전체 대비 정규화 영역 (x0, y0, x1, y1)
    width: int
    height: int
    mime: str
    data: bytes


@dataclass
class TileSet:
    """한 장을 타일로 나눈 결과. 내용이 없는 타일은 `tiles`에 넣지 않고 개수(`blank`)만 센다."""
    rows: int
    cols: int
    width: int                     # 타일을 나눈 기준 해상도(px)
    height: int
    tiles: list[RenderedTile]
    blank: int = 0
    dpi: float | None = None       # PDF 쪽일 때 실제로 렌더한 DPI


def is_blank_histogram(histogram: list[int]) -> bool:
    """밝기 히스토그램(256칸)으로 내용이 없는 타일인지 본다.

    가장 흔한 밝기를 배경으로 보고, 배경에서 TILE_BLANK_TOLERANCE보다 많이 벗어난 픽셀이
    TILE_BLANK_PIXEL_RATIO 이하면 빈 타일이다. 스캔본의 종이 질감 정도는 배경으로 친다.
    """
    total = sum(histogram)
    if total <= 0:
        return True
    background = max(range(len(histogram)), key=histogram.__getitem__)
    low = max(0, background - config.TILE_BLANK_TOLERANCE)
    high = min(len(histogram) - 1, background + config.TILE_BLANK_TOLERANCE)
    different = total - sum(histogram[low:high + 1])
    return different <= total * config.TILE_BLANK_PIXEL_RATIO
