"""Step 5 — 타일링: 분할 좌표·겹침, 타일 렌더, 좌표 역변환, 중복 박스·중복 줄 병합, 캐시 키, 모드 선택.

mock 모델은 **받은 이미지를 실제로 디코드해서** 그 안에 보이는 색 사각형에 따라 답한다.
그래서 "타일에 무엇이 담겼는가"와 "타일 좌표가 전체 좌표로 옳게 돌아오는가"를 끝에서 끝까지 확인할 수 있다.
"""
from __future__ import annotations

import io
import json

import pymupdf
import pytest
from PIL import Image, ImageChops

from app import config
from app.agent.grounding import merge_tile_boxes
from app.agent.prompts import TILE_GROUNDING_NOTE, TILE_OCR_NOTE
from app.agent.tools import ToolContext, execute_tool
from app.attachments import Attachment
from app.pipeline.geometry import needs_tiling, plan_tiles
from app.pipeline.images import TileSource, VisionUsage, assemble_model_images, crop_image_tiles
from app.pipeline.ocr import (OcrCache, TileText, build_ocr_reader, drop_overlap_duplicates, is_no_text_reply,
                              merge_tile_transcriptions, prepare_visual_ocr_evidence)
from app.pipeline.pdf import render_page_tiles
from app.providers.base import ModelResponse, Provider, ToolCall
from conftest import chat_body, upload
from mock_openai import all_text, image_count, is_grounding_call, is_ocr_call, request_images
from pdf_factory import (COLORS, build_drawing_pdf, build_pdf, build_scanned_pdf, color_box, colors_in, encode,
                         marked_image, png_bytes)

PDF = "application/pdf"
A3 = (1190, 842)          # pt. 200 DPI로 3306 x 2339 px → 2행 x 3열
A0 = (3370, 2384)         # pt. 200 DPI로 9361 x 6622 px → 5행 x 7열


def approx(box, **expected):
    return all(box[key] == pytest.approx(value, abs=1e-6) for key, value in expected.items())


class SeeingProvider(Provider):
    """받은 이미지를 보고 답하는 가짜 모델. 타일 호출은 동시에 나가므로 순서가 아니라 내용으로 답을 정한다."""
    name = "seeing"

    def __init__(self, reply):
        super().__init__(model="m")
        self.reply, self.calls = reply, []

    async def analyze(self, messages, images=None, tools=None, *, temperature=0.2, disable_thinking=False,
                      max_tokens=None):
        call = {"messages": messages, "images": images, "tools": tools, "temperature": temperature,
                "disable_thinking": disable_thinking, "max_tokens": max_tokens}
        self.calls.append(call)
        reply = self.reply(call)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, ModelResponse) else ModelResponse(text=reply)

    async def list_models(self):
        return ["m"]


def seen(call) -> Image.Image:
    (image,) = call["images"]
    return Image.open(io.BytesIO(image.data))


# --------------------------------------------------------------------------- 타일 분할 좌표·겹침
def test_tiles_cover_the_whole_area_with_the_configured_overlap():
    plan = plan_tiles(9362, 6622)                                   # A0 @ 200 DPI
    assert (plan.rows, plan.cols, len(plan.tiles), plan.scale) == (5, 7, 35, 1.0)
    assert {(tile.width, tile.height) for tile in plan.tiles} == {(1502, 1478)}      # 모든 타일이 같은 크기
    assert all(tile.width <= config.TILE_SIZE and tile.height <= config.TILE_SIZE for tile in plan.tiles)
    # 읽기 순서: 위 → 아래, 왼쪽 → 오른쪽
    assert [(tile.row, tile.col) for tile in plan.tiles] == [(r, c) for r in range(1, 6) for c in range(1, 8)]

    first_row = [tile for tile in plan.tiles if tile.row == 1]
    first_column = [tile for tile in plan.tiles if tile.col == 1]
    assert first_row[0].left == 0 and first_row[-1].right == 9362          # 가장자리까지 빠짐없이 덮는다
    assert first_column[0].top == 0 and first_column[-1].bottom == 6622
    expected = round(config.TILE_SIZE * config.TILE_OVERLAP)                # 1536 x 0.125 = 192px
    assert [a.right - b.left for a, b in zip(first_row, first_row[1:])] == [expected] * 6
    assert [a.bottom - b.top for a, b in zip(first_column, first_column[1:])] == [expected] * 4


def test_overlap_and_tile_size_come_from_config(monkeypatch):
    monkeypatch.setattr(config, "TILE_SIZE", 1000)
    monkeypatch.setattr(config, "TILE_OVERLAP", 0.2)
    plan = plan_tiles(2600, 900)
    assert (plan.rows, plan.cols) == (1, 3)
    row = list(plan.tiles)
    assert all(tile.width <= 1000 and tile.height == 900 for tile in row)
    assert all(a.right - b.left >= 200 for a, b in zip(row, row[1:]))
    assert plan_tiles(2600, 900, overlap=0.0).cols == 3
    assert plan_tiles(5000, 900, tile_size=2500, overlap=0.0).cols == 2


def test_source_boxes_are_fractions_of_the_whole():
    plan = plan_tiles(4000, 2000)
    boxes = [plan.source_box(tile) for tile in plan.tiles]
    assert boxes[0][:2] == (0.0, 0.0) and boxes[-1][2:] == (1.0, 1.0)
    assert boxes[-1] == pytest.approx((2538 / 4000, 904 / 2000, 1.0, 1.0))
    assert all(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 for x0, y0, x1, y1 in boxes)


def test_tile_count_is_capped_by_lowering_the_resolution():
    """A0를 400DPI로 나누면 140타일이다. 호출 수가 끝없이 늘지 않게 해상도를 낮춰 상한에 맞춘다."""
    plan = plan_tiles(18724, 13244)
    assert plan.scale < 1.0 and len(plan.tiles) <= config.MAX_TILES_PER_IMAGE == 48
    assert (plan.width, plan.height) == (round(18724 * plan.scale), round(13244 * plan.scale))
    assert max(max(tile.width, tile.height) for tile in plan.tiles) <= config.TILE_SIZE
    assert plan.source_box(plan.tiles[-1])[2:] == (1.0, 1.0)


def test_small_sources_are_not_tiled():
    assert not needs_tiling(2048, 1500) and not needs_tiling(1654, 1169)
    assert needs_tiling(2049, 100) and needs_tiling(3306, 2339)


def test_minimum_source_size_is_configurable(monkeypatch):
    monkeypatch.setattr(config, "TILE_MIN_SOURCE_EDGE", 1000)
    assert not needs_tiling(1536, 1536)          # 타일 한 장에 다 들어가면 나눌 이유가 없다
    assert needs_tiling(1537, 800)
    monkeypatch.setattr(config, "TILE_MIN_SOURCE_EDGE", 5000)
    assert not needs_tiling(4000, 3000) and needs_tiling(5001, 3000)


# --------------------------------------------------------------------------- PDF 타일
DRAWING_MARKS = [(100, 100, 300, 200, "red"), (900, 600, 1100, 800, "blue"), (500, 380, 700, 460, "green")]


def full_render(data: bytes, width: int, height: int) -> Image.Image:
    """비교용으로만 쪽 전체를 같은 배율로 그린다(실제 코드는 이렇게 하지 않는다)."""
    document = pymupdf.open(stream=data, filetype="pdf")
    page = document[0]
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(width / page.rect.width, height / page.rect.height), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    document.close()
    return image


@pytest.mark.parametrize("page_size,rotation", [(A3, 0), ((842, 1190), 90)])
def test_pdf_tiles_match_the_same_region_of_a_full_render(page_size, rotation):
    data = build_drawing_pdf(DRAWING_MARKS if not rotation else [(100, 100, 300, 300, "red"), (600, 900, 800, 1100, "blue")],
                             page_size=page_size, rotation=rotation)
    tiles = render_page_tiles(data, page_number=1)
    assert (tiles.rows, tiles.cols, tiles.width, tiles.height) == (2, 3, 3306, 2339)
    assert tiles.dpi == pytest.approx(200) and tiles.tiles
    whole = full_render(data, tiles.width, tiles.height)
    for tile in tiles.tiles:
        image = Image.open(io.BytesIO(tile.data)).convert("RGB")
        assert image.size == (tile.width, tile.height) and max(image.size) <= config.TILE_SIZE
        left, top = round(tile.box[0] * tiles.width), round(tile.box[1] * tiles.height)
        same_region = whole.crop((left, top, left + tile.width, top + tile.height))
        assert ImageChops.difference(same_region, image).getbbox() is None, (tile.row, tile.col)


def test_tiles_without_content_are_not_sent():
    tiles = render_page_tiles(build_drawing_pdf(DRAWING_MARKS), page_number=1)
    # 빨강은 r1c1, 파랑은 r2c3, 초록은 r1c2와 r2c2의 겹침 영역에 걸쳐 있다. r1c3·r2c1에는 아무것도 없다.
    assert [(tile.row, tile.col) for tile in tiles.tiles] == [(1, 1), (1, 2), (2, 2), (2, 3)] and tiles.blank == 2
    by_position = {(tile.row, tile.col): colors_in(Image.open(io.BytesIO(tile.data))) for tile in tiles.tiles}
    assert by_position == {(1, 1): ["red"], (1, 2): ["green"], (2, 2): ["green"], (2, 3): ["blue"]}

    sparse = render_page_tiles(build_drawing_pdf([(3000, 2000, 3200, 2200, "red")], page_size=A0), page_number=1)
    assert (sparse.rows, sparse.cols) == (5, 7) and len(sparse.tiles) + sparse.blank == 35
    assert len(sparse.tiles) <= 4 and sparse.blank >= 31                  # A0 한 장에 호출 35번이 아니라 몇 번


def test_blank_detection_ignores_paper_texture_but_keeps_faint_marks(monkeypatch):
    from app.pipeline.geometry import is_blank_histogram

    paper = [0] * 256
    paper[250], paper[243], paper[255] = 900_000, 60_000, 40_000        # 스캔본의 종이 질감: 배경 주변에 몰려 있다
    assert is_blank_histogram(paper)
    inked = list(paper)
    inked[40] = 300                                                      # 작은 글자 하나 분량의 어두운 픽셀
    assert not is_blank_histogram(inked)
    monkeypatch.setattr(config, "TILE_BLANK_PIXEL_RATIO", 0.0)
    speck = list(paper)
    speck[0] = 1
    assert not is_blank_histogram(speck) and is_blank_histogram(paper)


def scanned_a3(width: int, height: int) -> bytes:
    """A3 한 장을 덮는 스캔 이미지. width 2480 → 150 DPI, 3306 → 200 DPI, 4958 → 300 DPI."""
    marks = [(width // 10, height // 10, width // 5, height // 5, "red"),
             (width * 8 // 10, height * 8 // 10, width * 9 // 10, height * 9 // 10, "blue")]
    return build_scanned_pdf(encode(marked_image(width, height, marks)), page_size=A3)


def test_scanned_pages_are_rendered_no_finer_than_the_embedded_image():
    """스캔본은 박힌 이미지의 원래 해상도까지만. 그보다 높이면 픽셀만 늘고 선명해지지 않는다."""
    coarse = render_page_tiles(scanned_a3(2480, 1754), page_number=1)          # 150 DPI 스캔
    assert coarse.dpi == pytest.approx(150, abs=0.2) and coarse.width == 2480
    exact = render_page_tiles(scanned_a3(3306, 2339), page_number=1)           # 200 DPI 스캔
    assert exact.dpi == pytest.approx(200, abs=0.2) and exact.width == 3306
    fine = render_page_tiles(scanned_a3(4958, 3508), page_number=1)            # 300 DPI 스캔 → 설정값(200)까지만
    assert fine.dpi == pytest.approx(config.TILE_RENDER_DPI) and fine.width == 3306
    # 해상도가 낮은 스캔은 나눌 만큼 크지 않다 → 전체 모드와 같게 처리된다.
    assert render_page_tiles(scanned_a3(1654, 1169), page_number=1) is None     # 100 DPI → 1654px


def test_vector_pages_use_the_configured_tile_dpi(monkeypatch):
    data = build_drawing_pdf(DRAWING_MARKS)
    assert render_page_tiles(data, page_number=1).dpi == pytest.approx(200)
    monkeypatch.setattr(config, "TILE_RENDER_DPI", 300)
    sharper = render_page_tiles(data, page_number=1)
    assert sharper.dpi == pytest.approx(300) and sharper.width == round(1190 * 300 / 72)

    # 래스터와 벡터가 섞인 쪽: 선은 DPI만큼 선명해지므로 박힌 이미지 해상도로 제한하지 않는다.
    document = pymupdf.open()
    page = document.new_page(width=A3[0], height=A3[1])
    page.insert_image(page.rect, stream=encode(marked_image(1240, 877, [(100, 100, 300, 300, "red")])))   # 75 DPI
    page.draw_line((50, 50), (1100, 800), color=(0, 0, 0), width=0.5)
    mixed = document.tobytes()
    document.close()
    assert render_page_tiles(mixed, page_number=1).dpi == pytest.approx(300)


def test_pdf_tile_errors_are_user_facing():
    from app.pipeline.pdf import PdfError
    with pytest.raises(PdfError, match="범위"):
        render_page_tiles(build_drawing_pdf(DRAWING_MARKS), page_number=5)


# --------------------------------------------------------------------------- 이미지 타일(보관해 둔 원본에서)
def test_image_tiles_are_cut_from_the_original_at_full_resolution():
    original = encode(marked_image(4000, 2000, [(3000, 1500, 3200, 1700, "red")]))
    tiles = crop_image_tiles(original, "image/png")
    assert (tiles.rows, tiles.cols, tiles.width, tiles.height) == (2, 3, 4000, 2000)
    assert [(tile.row, tile.col) for tile in tiles.tiles] == [(2, 3)] and tiles.blank == 5
    (tile,) = tiles.tiles
    image = Image.open(io.BytesIO(tile.data))
    assert image.size == (tile.width, tile.height) == (1462, 1096) and tile.mime == "image/png"
    assert tile.box == pytest.approx((2538 / 4000, 904 / 2000, 1.0, 1.0))
    # 줄이지 않았다: 원본의 200px 사각형이 타일에서도 200px이다(3072px 사본에서는 154px).
    assert color_box(image, "red") == (3000 - 2538, 1500 - 904, 3200 - 2538, 1700 - 904)


def test_image_tiles_follow_the_upright_orientation_and_source_format():
    sideways = marked_image(4000, 2000, [(100, 100, 400, 300, "blue")])
    exif = Image.Exif()
    exif[0x0112] = 6                                                 # 시계 방향 90° 회전해서 봐야 하는 사진
    tiles = crop_image_tiles(encode(sideways, "JPEG", quality=95, exif=exif), "image/jpeg")
    assert (tiles.width, tiles.height, tiles.rows, tiles.cols) == (2000, 4000, 3, 2)
    assert {tile.mime for tile in tiles.tiles} == {"image/jpeg"}
    assert [(tile.row, tile.col) for tile in tiles.tiles] == [(1, 2)]          # 왼쪽 위 → 회전 후 오른쪽 위

    # 투명 배경(픽셀 값은 검정) 위의 불투명한 표시: 사본과 똑같이 흰 배경으로 깔고 자른다.
    transparent = Image.new("RGBA", (3000, 1200), (0, 0, 0, 0))
    transparent.paste((*COLORS["green"], 255), (2500, 500, 2700, 700))
    cut = crop_image_tiles(encode(transparent), "image/png")
    assert (cut.rows, cut.cols, cut.blank) == (1, 3, 2) and [(tile.row, tile.col) for tile in cut.tiles] == [(1, 3)]
    flattened = Image.open(io.BytesIO(cut.tiles[0].data))
    assert flattened.mode == "RGB" and flattened.getpixel((0, 0)) == (255, 255, 255)
    assert color_box(flattened, "green") == (2500 - 1872, 500, 2700 - 1872, 700)

    assert crop_image_tiles(png_bytes(1600, 1200), "image/png") is None


def test_image_tile_count_is_capped_too(monkeypatch):
    monkeypatch.setattr(config, "MAX_TILES_PER_IMAGE", 2)
    marks = [(200, 200, 600, 600, "red"), (3400, 1400, 3800, 1800, "blue")]
    tiles = crop_image_tiles(encode(marked_image(4000, 2000, marks)), "image/png")
    assert tiles.rows * tiles.cols <= 2 and tiles.width < 4000
    assert all(max(tile.width, tile.height) <= config.TILE_SIZE for tile in tiles.tiles)
    assert tiles.tiles[0].box[:2] == (0.0, 0.0) and tiles.tiles[-1].box[2:] == (1.0, 1.0)
    assert [colors_in(Image.open(io.BytesIO(tile.data))) for tile in tiles.tiles] == [["red"], ["blue"]]


# --------------------------------------------------------------------------- 모드가 갈리는 한 곳
async def test_mode_branches_only_in_assemble_model_images():
    original = encode(marked_image(4000, 2000, [(3000, 1500, 3200, 1700, "red"), (100, 100, 300, 300, "blue")]))
    source = TileSource(kind="image", data=original, mime="image/png")
    copy = b"3072px-copy"

    for purpose in ("ocr", "grounding", "analysis"):                        # 전체 모드: 원본이 있어도 한 장
        (whole,) = await assemble_model_images("plan.png", "image/png", copy, purpose=purpose, source=source)
        assert whole.data == copy and whole.source_box == (0.0, 0.0, 1.0, 1.0) and whole.grid is None

    # 타일 모드여도 답변(analysis) 호출은 전체 한 장이다 — 추론 호출의 이미지는 Step 8에서 다룬다.
    (analysis,) = await assemble_model_images("plan.png", "image/png", copy, purpose="analysis", mode="tile", source=source)
    assert analysis.data == copy and analysis.grid is None

    for purpose in ("ocr", "grounding"):
        tiles = await assemble_model_images("plan.png", "image/png", copy, purpose=purpose, mode="tile", source=source)
        assert [tile.tile for tile in tiles] == [(1, 1), (2, 3)]
        assert [tile.name for tile in tiles] == ["plan.png · tile r1c1", "plan.png · tile r2c3"]
        grid = tiles[0].grid
        assert grid is tiles[1].grid and (grid.rows, grid.cols, grid.blank) == (2, 3, 4)
        assert all(tile.data != copy and (tile.width, tile.height) == (1462, 1096) for tile in tiles)


async def test_small_or_missing_sources_are_handled_exactly_like_whole_mode():
    copy = png_bytes(1600, 1200)
    small = TileSource(kind="image", data=copy, mime="image/png")
    blank = TileSource(kind="image", data=encode(marked_image(4000, 2000, [])), mime="image/png")
    native_pdf = TileSource(kind="pdf", data=build_pdf("native"), page_number=1)      # A4 @ 200 DPI = 2339px
    (expected,) = await assemble_model_images("a.png", "image/png", copy, purpose="ocr")
    for source in (small, blank, None, TileSource(kind="image", data=b"", mime="image/png")):
        assert await assemble_model_images("a.png", "image/png", copy, purpose="ocr", mode="tile", source=source) == [expected]
    tiles = await assemble_model_images("a.pdf · page 1", "image/png", copy, purpose="ocr", mode="tile", source=native_pdf)
    assert len(tiles) > 1 and tiles[0].grid.dpi == pytest.approx(200)


# --------------------------------------------------------------------------- bbox: 좌표 역변환과 중복 병합
def box_reply(image: Image.Image, colors=("red", "green", "blue"), pixels: bool = False) -> str:
    """가짜 grounding 모델: 받은 이미지(타일)에서 색 사각형을 찾아 **그 이미지 기준** 좌표로 답한다."""
    regions = []
    for color in colors:
        found = color_box(image, color)
        if found is None:
            continue
        if pixels:
            regions.append({"type": "object", "label": color, "bbox_2d": list(found)})
        else:
            scaled = [round(found[0] * 1000 / image.width), round(found[1] * 1000 / image.height),
                      round(found[2] * 1000 / image.width), round(found[3] * 1000 / image.height)]
            regions.append({"type": "object", "label": color, "bbox": scaled, "confidence": 0.9})
    return json.dumps({"text": ", ".join(region["label"] for region in regions), "regions": regions})


def plan_image(marks, width=4000, height=2000) -> tuple[Attachment, bytes]:
    """업로드 이미지 첨부: data는 3072px 사본 자리, source_data가 원본."""
    original = encode(marked_image(width, height, marks))
    surface = Attachment(name="plan.png", kind="image", mime="image/png", data=png_bytes(3072, 1536),
                         width=3072, height=1536, source_data=original, source_mime="image/png")
    return surface, original


async def test_tile_boxes_are_mapped_back_to_the_full_image():
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red"), (200, 300, 600, 500, "blue")])
    provider = SeeingProvider(lambda call: box_reply(seen(call)))
    context = ToolContext(provider=provider, attachments=[surface], image_mode="tile")
    output = json.loads(await execute_tool(context, ToolCall("inspect_visual", {"name": "plan.png", "task": "find squares"})))

    # 타일 두 장(r1c1, r2c3)만 모델에 갔고, 각 타일의 좌표가 원본 4000 x 2000 기준으로 돌아왔다.
    assert len(provider.calls) == 2 and "warning" not in output
    regions = {region["label"]: region for region in output["regions"]}
    assert regions["red"]["x"] == pytest.approx(3000 / 4000, abs=0.002) and regions["red"]["y"] == pytest.approx(0.75, abs=0.002)
    assert regions["red"]["w"] == pytest.approx(0.05, abs=0.002) and regions["red"]["h"] == pytest.approx(0.1, abs=0.002)
    assert regions["blue"]["x"] == pytest.approx(0.05, abs=0.002) and regions["blue"]["y"] == pytest.approx(0.15, abs=0.002)
    assert regions["blue"]["w"] == pytest.approx(0.1, abs=0.002) and regions["blue"]["h"] == pytest.approx(0.1, abs=0.002)
    assert context.artifacts[0]["boxes"] == output["regions"] and context.artifacts[0]["name"] == "plan.png"
    # 타일마다 분리된 호출: 전용 프롬프트, temperature 0, 이미지 한 장, 타일이라는 안내
    for call in provider.calls:
        assert "visual grounding engine" in call["messages"][0]["content"] and call["temperature"] == 0.0
        assert len(call["images"]) == 1 and call["tools"] is None
        assert TILE_GROUNDING_NOTE in call["messages"][1]["content"]
    assert (context.usage.grounding_calls, context.usage.tiles, context.usage.blank_tiles) == (2, 2, 4)


async def test_pixel_coordinates_are_divided_by_the_tile_size_not_the_page_size():
    """픽셀 좌표로 답하는 모델(Qwen2.5-VL 등): 모델이 본 것은 타일이므로 타일 크기로 나눠야 한다."""
    surface, _ = plan_image([(3700, 1500, 3900, 1700, "red")])
    provider = SeeingProvider(lambda call: box_reply(seen(call), pixels=True))
    context = ToolContext(provider=provider, attachments=[surface], image_mode="tile")
    output = json.loads(await execute_tool(context, ToolCall("inspect_visual", {"name": "plan.png", "task": "find"})))
    (call,) = provider.calls
    assert color_box(seen(call), "red") == (1162, 596, 1362, 796)      # 타일(1462 x 1096) 안의 픽셀 좌표
    (red,) = output["regions"]
    assert red["x"] == pytest.approx(3700 / 4000, abs=0.002) and red["y"] == pytest.approx(1500 / 2000, abs=0.002)
    assert red["w"] == pytest.approx(200 / 4000, abs=0.002) and red["h"] == pytest.approx(200 / 2000, abs=0.002)


def test_pixel_coordinates_below_1000_cannot_be_told_from_the_requested_frame():
    """알려진 한계(기존 파서 규칙 그대로): 값이 모두 1000 이하면 요청한 0~1000 좌표로 읽는다.

    타일은 작아서(≤ 1536px) 픽셀로 답하는 모델의 좌표가 모두 1000 이하인 경우가 전체 모드보다 흔하다.
    """
    from app.agent.grounding import parse_visual_inspection
    parsed = parse_visual_inspection('[{"label":"x","bbox_2d":[462,596,662,796]}]', image_width=1462, image_height=1096)
    assert approx(parsed.boxes[0], x=0.462, y=0.596, w=0.2, h=0.2)       # 픽셀이었다면 x=0.316이어야 한다


async def test_whole_mode_grounding_is_unchanged_by_tiling():
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red")])
    provider = SeeingProvider(lambda call: '{"text":"","regions":[{"type":"object","label":"red","bbox":[750,750,800,850]}]}')
    context = ToolContext(provider=provider, attachments=[surface])                 # image_mode 기본값 = whole
    output = json.loads(await execute_tool(context, ToolCall("inspect_visual", {"name": "plan.png", "task": "find"})))
    (call,) = provider.calls
    assert call["images"][0].data == surface.data and call["images"][0].grid is None     # 사본 한 장 그대로
    assert TILE_GROUNDING_NOTE not in call["messages"][1]["content"]
    assert approx(output["regions"][0], x=0.75, y=0.75, w=0.05, h=0.1)
    assert context.usage.tiles == 0 and context.usage.grounding_calls == 1


async def test_target_in_the_overlap_is_found_twice_and_merged_into_one_box():
    # 가로 겹침 영역(2538~2731px)에 온전히 들어가는 대상 → r1c2와 r1c3가 둘 다 본다.
    surface, _ = plan_image([(2560, 300, 2700, 440, "green"), (3500, 300, 3700, 500, "red")])
    provider = SeeingProvider(lambda call: box_reply(seen(call)))
    context = ToolContext(provider=provider, attachments=[surface], image_mode="tile")
    output = json.loads(await execute_tool(context, ToolCall("inspect_visual", {"name": "plan.png", "task": "find"})))
    assert len(provider.calls) == 2
    assert sorted(region["label"] for region in output["regions"]) == ["green", "red"]   # green은 두 번 잡혔지만 하나로
    green = next(region for region in output["regions"] if region["label"] == "green")
    assert green["x"] == pytest.approx(2560 / 4000, abs=0.002) and green["w"] == pytest.approx(140 / 4000, abs=0.003)


def box(x, y, w, h, kind="object", label="", confidence=None):
    item = {"x": x, "y": y, "w": w, "h": h, "type": kind, "label": label}
    if confidence is not None:
        item["confidence"] = confidence
    return item


def test_duplicate_boxes_from_neighbouring_tiles_are_merged():
    # 같은 대상을 두 타일이 거의 같게 잡았다(IoU ≥ 0.5) → 하나. 라벨은 더 믿을 만한 쪽.
    merged = merge_tile_boxes([[box(0.40, 0.10, 0.10, 0.10, "stamp", "APPROVED", 0.6)],
                               [box(0.41, 0.10, 0.10, 0.11, "stamp", "APPROVED stamp", 0.9)]])
    assert len(merged) == 1 and merged[0]["label"] == "APPROVED stamp" and merged[0]["confidence"] == 0.9
    assert approx(merged[0], x=0.40, y=0.10, w=0.11, h=0.11)                       # 두 박스를 모두 덮는다

    # 한 타일은 가장자리에서 잘린 일부만 봤다(작은 박스가 큰 박스 안에 있다) → 하나.
    (partial,) = merge_tile_boxes([[box(0.30, 0.20, 0.20, 0.10, "table", "BOM")], [box(0.44, 0.20, 0.06, 0.10, "table", "BOM")]])
    assert approx(partial, x=0.30, y=0.20, w=0.20, h=0.10)

    # 세 타일이 본 같은 대상도 하나가 된다.
    assert len(merge_tile_boxes([[box(0.5, 0.5, 0.1, 0.1)], [box(0.5, 0.5, 0.1, 0.1)], [box(0.51, 0.5, 0.1, 0.1)]])) == 1


def test_boxes_that_are_different_targets_are_kept_apart():
    # 같은 타일 안의 박스는 겹쳐도 합치지 않는다(모델이 한 이미지에서 따로 잡은 것은 전체 모드처럼 그대로).
    assert len(merge_tile_boxes([[box(0.1, 0.1, 0.2, 0.2), box(0.1, 0.1, 0.2, 0.2)]])) == 2
    # 종류가 다르면 다른 대상이다: 표 안의 글자.
    assert len(merge_tile_boxes([[box(0.1, 0.1, 0.4, 0.4, "table")], [box(0.2, 0.2, 0.1, 0.05, "text")]])) == 2
    # 조금 겹칠 뿐인 이웃 대상(IoU 0.14, 포함 비율 0.25)은 합치지 않는다.
    assert len(merge_tile_boxes([[box(0.10, 0.1, 0.2, 0.1)], [box(0.25, 0.1, 0.2, 0.1)]])) == 2
    # 떨어져 있는 같은 라벨(창호 기호 W1이 여러 개)도 그대로.
    assert len(merge_tile_boxes([[box(0.1, 0.1, 0.05, 0.05, "object", "W1")], [box(0.6, 0.1, 0.05, 0.05, "object", "W1")]])) == 2
    assert merge_tile_boxes([]) == [] and merge_tile_boxes([[], []]) == []


def test_merge_thresholds_come_from_config(monkeypatch):
    pair = [[box(0.10, 0.1, 0.2, 0.1)], [box(0.25, 0.1, 0.2, 0.1)]]
    monkeypatch.setattr(config, "TILE_BOX_MERGE_CONTAINMENT", 0.2)
    assert len(merge_tile_boxes(pair)) == 1


async def test_each_tile_is_retried_on_its_own_and_partial_failures_are_reported():
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red"), (200, 300, 600, 500, "blue")])
    attempts: dict[tuple, int] = {}

    def reply(call):
        image = call["images"][0]
        attempts[image.tile] = attempts.get(image.tile, 0) + 1
        if image.tile == (1, 1):
            return "The blue square is in the upper left."                       # 끝까지 산문으로만 답하는 타일
        return "not json" if attempts[image.tile] == 1 else box_reply(seen(call))

    provider = SeeingProvider(reply)
    context = ToolContext(provider=provider, attachments=[surface], image_mode="tile")
    output = json.loads(await execute_tool(context, ToolCall("inspect_visual", {"name": "plan.png", "task": "find"})))
    assert attempts == {(1, 1): 1 + config.GROUNDING_RETRY_COUNT, (2, 3): 2}       # 실패한 타일만 다시 묻는다
    assert [region["label"] for region in output["regions"]] == ["red"]
    assert output["warning"].startswith("1 of 2 tiles did not return structured regions")
    retried = [call for call in provider.calls if "Return only the JSON object" in call["messages"][1]["content"]]
    assert len(retried) == config.GROUNDING_RETRY_COUNT + 1

    # 어느 타일도 구조화된 답을 내지 못하면 전체 모드와 같은 경고.
    nothing = ToolContext(provider=SeeingProvider(lambda call: "prose only"), attachments=[plan_image(
        [(3000, 1500, 3200, 1700, "red")])[0]], image_mode="tile")
    failed = json.loads(await execute_tool(nothing, ToolCall("inspect_visual", {"name": "plan.png", "task": "find"})))
    assert failed["regions"] == [] and "did not return structured regions; no boxes" in failed["warning"]
    assert "boxes" not in nothing.artifacts[0]


async def test_pdf_page_bbox_is_tiled_from_the_pdf_itself():
    pdf = Attachment(name="plan.pdf", kind="pdf", mime="application/pdf", data=build_drawing_pdf(DRAWING_MARKS), text="t")
    provider = SeeingProvider(lambda call: box_reply(seen(call)))
    context = ToolContext(provider=provider, attachments=[pdf], image_mode="tile")
    output = json.loads(await execute_tool(context, ToolCall("inspect_visual", {"name": "plan.pdf", "page": 1, "task": "find"})))
    assert output["source"] == "plan.pdf · page 1" and len(provider.calls) == 4
    assert all(max(seen(call).size) <= config.TILE_SIZE for call in provider.calls)
    regions = {region["label"]: region for region in output["regions"]}
    assert sorted(regions) == ["blue", "green", "red"]                    # 초록은 세로 겹침에서 두 번 잡혀 하나로 합쳐졌다
    assert regions["red"]["x"] == pytest.approx(100 / 1190, abs=0.002) and regions["red"]["w"] == pytest.approx(200 / 1190, abs=0.002)
    assert regions["green"]["y"] == pytest.approx(380 / 842, abs=0.002) and regions["green"]["h"] == pytest.approx(80 / 842, abs=0.003)
    assert regions["blue"]["y"] == pytest.approx(600 / 842, abs=0.002)


# --------------------------------------------------------------------------- OCR 병합
def part(row, col, text, status="ok", size=0.4):
    left, top = (col - 1) * 0.3, (row - 1) * 0.3
    return TileText(row=row, col=col, box=(left, top, left + size, top + size), text=text, status=status)


def test_tile_transcriptions_are_joined_in_reading_order():
    parts = [part(2, 1, "BOTTOM LEFT NOTE"), part(1, 2, "TOP RIGHT TITLE"), part(1, 1, "TOP LEFT HEADER"), part(2, 2, "")]
    merged = merge_tile_transcriptions(parts, rows=2, cols=2, blank=3)
    assert merged.index("TOP LEFT HEADER") < merged.index("TOP RIGHT TITLE") < merged.index("BOTTOM LEFT NOTE")
    assert merged.splitlines()[0].startswith("[TILED TRANSCRIPTION: this page was read as 2 rows x 2 columns")
    assert "3 tiles contain text, 4 contain none" in merged.splitlines()[0]      # 글자 없는 1장 + 빈 타일 3장
    assert "[TILE r1c1]\nTOP LEFT HEADER\n[TILE r1c2]\nTOP RIGHT TITLE\n[TILE r2c1]\nBOTTOM LEFT NOTE" in merged
    assert "[TILE r2c2]" not in merged


def test_duplicate_lines_in_the_overlap_are_removed_once():
    left = part(1, 1, "DWG NO: FA-7731-B\nSEE DETAIL A-A\n  ITEM   PART NO     QTY")
    right = part(1, 2, "SEE  DETAIL A-A\nSCALE 1:5\n  ITEM   PART NO     QTY")
    first, second = drop_overlap_duplicates([right, left])
    assert first.text == left.text                                  # 앞 타일은 그대로
    assert second.text == "SCALE 1:5"                               # 뒤 타일에서만, 공백 차이는 무시하고 지운다

    # 겹치지 않는 타일(이웃이 아님)의 같은 줄은 문서에 두 번 있는 것이다.
    far = TileText(row=1, col=3, box=(0.8, 0.0, 1.0, 0.4), text="SEE DETAIL A-A")
    assert drop_overlap_duplicates([left, far])[1].text == "SEE DETAIL A-A"


def test_repeated_document_content_is_never_removed():
    """CLAUDE.md: 전사 정리는 잡음만 지운다. 같은 값이 이어지는 표, 짧은 값, 숫자는 문서 내용이다."""
    table = "QTY\n2\n2\n2\nM12\n1200\n2026-03-14\n[UNCLEAR]"          # 두 타일에 똑같이 찍힌 줄들
    above, below = part(1, 1, f"BOLT M12x40 8.8 ZN\n{table}"), part(2, 1, f"BOLT M12x40 8.8 ZN\n{table}")
    _, kept = drop_overlap_duplicates([above, part(2, 1, table)])
    assert kept.text == table                                       # 짧은 값·숫자·날짜·표식은 그대로

    # 한 타일 안에서 되풀이되는 긴 줄(같은 행이 이어지는 표)도 그대로 둔다.
    rows = "WASHER 12 ZN PLATED\nWASHER 12 ZN PLATED\nWASHER 12 ZN PLATED"
    _, repeated = drop_overlap_duplicates([part(1, 1, "WASHER 12 ZN PLATED"), part(2, 1, rows)])
    assert repeated.text == rows
    _, unique_below = drop_overlap_duplicates([part(1, 1, rows), part(2, 1, "WASHER 12 ZN PLATED")])
    assert unique_below.text == "WASHER 12 ZN PLATED"
    # 길고 글자가 든 줄 하나만 겹침 중복으로 지워진다.
    assert [item.text for item in drop_overlap_duplicates([above, below])] == [above.text, table]


def test_a_removed_line_does_not_cascade_to_the_next_tile():
    """r1c1·r1c2·r1c3에 같은 줄이 하나씩: 가운데 것은 왼쪽과의 중복으로 지워지고, 오른쪽 것은 남는다."""
    parts = drop_overlap_duplicates([part(1, 1, "GENERAL NOTES"), part(1, 2, "GENERAL NOTES"), part(1, 3, "GENERAL NOTES")])
    assert [item.text for item in parts] == ["GENERAL NOTES", "", "GENERAL NOTES"]


def test_tiles_without_text_and_failed_tiles_are_reported_honestly():
    for reply in ("[NO TEXT]", "[no text]", "NO TEXT", " [NO TEXT]. ", "No text"):
        assert is_no_text_reply(reply), reply
    for reply in ("", "NO TEXT VISIBLE IN SECTION B", "[UNCLEAR]", "NOTE: NO TEXT"):
        assert not is_no_text_reply(reply), reply

    failure = "[OCR FAILED AFTER 3 ATTEMPTS: empty or non-transcription OCR response]"
    merged = merge_tile_transcriptions([part(1, 1, "TITLE BLOCK REV C"), part(1, 2, failure, status="failed"),
                                        part(2, 1, "", status="empty")], rows=2, cols=2, blank=1)
    assert "1 tiles contain text, 2 contain none, 1 could not be read" in merged
    assert f"[TILE r1c2]\n{failure}" in merged and "[TILE r2c1]" not in merged
    # 한 타일도 읽지 못했으면 쪽 전체의 전사 실패다(전체 모드와 같은 표식 → 캐시하지 않는다).
    assert merge_tile_transcriptions([part(1, 1, failure, status="failed")], rows=1, cols=2) == failure


RED_TEXT, GREEN_TEXT, BLUE_TEXT = "DWG NO: FA-7731-B", "SEE DETAIL A-A", "APPROVED BY S. LEE"
TEXT_BY_COLOR = {"red": RED_TEXT, "green": GREEN_TEXT, "blue": BLUE_TEXT}
# A3 @ 200 DPI(3306 x 2339): 빨강은 r1c1에만, 초록은 r1c1과 r1c2의 겹침 영역에, 파랑은 r2c3에만 있다.
SCAN_MARKS = [(200, 200, 400, 400, "red"), (1080, 300, 1200, 420, "green"), (2900, 1900, 3100, 2100, "blue")]


def transcribe(image: Image.Image) -> str:
    """가짜 전사 모델: 보이는 색마다 정해진 글자를 '읽는다'. 아무것도 없으면 약속된 표식으로 답한다."""
    return "\n".join(TEXT_BY_COLOR[color] for color in colors_in(image)) or "[NO TEXT]"


def scanned_drawing() -> bytes:
    return build_scanned_pdf(encode(marked_image(3306, 2339, SCAN_MARKS)), page_size=A3)


def page_of(pdf: bytes, number: int = 1) -> list[Attachment]:
    root = Attachment(name="scan.pdf", kind="pdf", mime="application/pdf", data=pdf, text="native")
    image = Attachment(name=f"scan.pdf · page {number}", kind="image", mime="image/png", data=png_bytes(3072, 2173),
                       has_data=True, page_number=number, page_classification="scanned-raster", ocr_required=True)
    return [root, image]


def tile_loader(attachments):
    async def load(page):
        return TileSource(kind="pdf", data=attachments[0].data, page_number=page.page_number)
    return load


async def test_tile_mode_transcribes_each_tile_and_merges_the_page():
    attachments = page_of(scanned_drawing())
    provider = SeeingProvider(lambda call: transcribe(seen(call)))
    usage, progress = VisionUsage(), []
    reader = build_ocr_reader(provider, image_mode="tile", load_tile_source=tile_loader(attachments), usage=usage,
                              on_progress=progress.append)
    text = await reader(attachments[1], "INSTRUCTION")

    assert len(provider.calls) == 3                                   # 6타일 중 내용이 있는 3장만(r1c1, r1c2, r2c3)
    assert (usage.ocr_calls, usage.tiled_images, usage.tiles, usage.blank_tiles) == (3, 1, 3, 3)
    assert text.splitlines()[1:] == ["[TILE r1c1]", RED_TEXT, GREEN_TEXT, "[TILE r2c3]", BLUE_TEXT]
    assert text.count(GREEN_TEXT) == 1                                # 겹침 영역의 글자는 한 번만
    assert "2 tiles contain text, 4 contain none" in text
    for call in provider.calls:
        assert "transcription engine" in call["messages"][0]["content"] and call["temperature"] == 0.0
        assert call["messages"][1]["content"] == f"INSTRUCTION\n{TILE_OCR_NOTE}"
        assert max(seen(call).size) <= config.TILE_SIZE and call["images"][0].mime == "image/png"
    assert progress[-1] == "타일 전사 중… scan.pdf · page 1 (3/3)"


async def test_whole_mode_reader_is_unchanged_by_tiling():
    attachments = page_of(scanned_drawing())
    provider = SeeingProvider(lambda call: "WHOLE PAGE TEXT")
    usage = VisionUsage()
    text = await build_ocr_reader(provider, usage=usage, load_tile_source=tile_loader(attachments))(attachments[1], "INSTRUCTION")
    (call,) = provider.calls
    assert text == "WHOLE PAGE TEXT" and call["images"][0].data == attachments[1].data
    assert call["messages"][1]["content"] == "INSTRUCTION" and usage.tiles == 0 and usage.ocr_calls == 1


async def test_tile_retries_and_failures_stay_inside_the_tile():
    attachments = page_of(scanned_drawing())
    attempts: dict[tuple, int] = {}

    def reply(call):
        tile = call["images"][0].tile
        attempts[tile] = attempts.get(tile, 0) + 1
        if tile == (2, 3):
            return "The image shows a blue square."                  # 끝까지 설명만 하는 타일
        if tile == (1, 2) and attempts[tile] == 1:
            return RuntimeError("timeout")
        return transcribe(seen(call))

    provider = SeeingProvider(reply)
    reader = build_ocr_reader(provider, image_mode="tile", load_tile_source=tile_loader(attachments))
    text = await reader(attachments[1], "INSTRUCTION")
    assert attempts == {(1, 1): 1, (1, 2): 2, (2, 3): config.OCR_RETRY_COUNT}
    assert RED_TEXT in text and "[TILE r2c3]\n[OCR FAILED AFTER 3 ATTEMPTS" in text and "1 could not be read" in text
    retried = [call for call in provider.calls if "previous attempt was not a valid transcription" in call["messages"][1]["content"]]
    assert len(retried) == 1 + (config.OCR_RETRY_COUNT - 1)

    # 타일 하나라도 실패한 쪽은 캐시하지 않는다 → 다음에 다시 시도한다.
    cache, calls = OcrCache(), []

    async def read(attachment, _instruction):
        calls.append(attachment.name)
        return text

    for _ in range(2):
        pages = page_of(scanned_drawing())
        await prepare_visual_ocr_evidence(pages, read_image=read, cache=cache, image_mode="tile")
    assert len(calls) == 2 and len(cache) == 0


async def test_tiles_never_exceed_the_configured_concurrency():
    import asyncio
    running = peak = 0

    class Slow(SeeingProvider):
        async def analyze(self, messages, images=None, tools=None, *, temperature=0.2, **_options):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            return ModelResponse(text="TEXT LINE")

    marks = [(x, y, x + 100, y + 100, "red") for x in range(100, 9000, 700) for y in range(100, 6000, 700)]
    pdf = build_drawing_pdf([(x / 2.78, y / 2.78, (x + 100) / 2.78, (y + 100) / 2.78, c) for x, y, _, _, c in marks], page_size=A0)
    pages = [Attachment(name="big.pdf", kind="pdf", mime="application/pdf", data=pdf)]
    pages += [Attachment(name=f"big.pdf · page {n}", kind="image", mime="image/png", data=f"p{n}".encode(), has_data=True,
                         page_number=1, ocr_required=True) for n in (1, 2)]
    provider = Slow(lambda call: "")
    reader = build_ocr_reader(provider, image_mode="tile", load_tile_source=tile_loader(pages))
    await prepare_visual_ocr_evidence(pages, read_image=reader, cache=OcrCache(), image_mode="tile")
    assert peak == config.OCR_CONCURRENCY == 2                        # 쪽 2장 x 타일 35장이어도 동시 호출은 2


async def test_tile_problems_fall_back_to_the_whole_image():
    attachments = page_of(b"%PDF-1.4 broken")
    provider = SeeingProvider(lambda call: "WHOLE PAGE TEXT")
    progress = []
    reader = build_ocr_reader(provider, image_mode="tile", load_tile_source=tile_loader(attachments),
                              on_progress=progress.append)
    assert await reader(attachments[1], "INSTRUCTION") == "WHOLE PAGE TEXT"
    assert provider.calls[0]["images"][0].data == attachments[1].data
    assert any("전체 이미지로 전사" in message for message in progress)


# --------------------------------------------------------------------------- 캐시 키 · 모드 전환
def ocr_page(number: int, data: bytes) -> Attachment:
    return Attachment(name=f"drawing.pdf · page {number}", kind="image", mime="image/png", data=data, has_data=True,
                      page_number=number, page_classification="scanned-raster", ocr_required=True)


def test_cache_key_contains_the_mode_and_tile_settings(monkeypatch):
    whole = OcrCache.key("m", "image/png", b"page", config.image_mode_variant("whole"))
    tile = OcrCache.key("m", "image/png", b"page", config.image_mode_variant("tile"))
    assert whole != tile and whole == OcrCache.key("m", "image/png", b"page")
    assert config.image_mode_variant("whole") == "whole"
    before = config.image_mode_variant("tile")
    for name, value in (("TILE_SIZE", 1024), ("TILE_OVERLAP", 0.25), ("TILE_RENDER_DPI", 300),
                        ("TILE_MIN_SOURCE_EDGE", 3072), ("MAX_TILES_PER_IMAGE", 24)):
        monkeypatch.setattr(config, name, value)
        changed = config.image_mode_variant("tile")
        assert changed != before, name
        before = changed
    assert config.image_mode_variant("whole") == "whole"              # 타일 설정은 전체 모드의 키를 바꾸지 않는다


async def test_cached_results_are_not_reused_across_modes(monkeypatch):
    cache, calls = OcrCache(), []

    async def read(attachment, _instruction):
        calls.append(attachment.name)
        return f"text #{len(calls)}"

    async def run(mode):
        output, _ = await prepare_visual_ocr_evidence([ocr_page(1, b"same-bytes")], read_image=read, cache=cache,
                                                      cache_namespace="provider:url:model", image_mode=mode)
        return output[-1].text.splitlines()[-1]

    assert [await run(mode) for mode in ("whole", "tile", "whole", "tile")] == ["text #1", "text #2", "text #1", "text #2"]
    monkeypatch.setattr(config, "TILE_SIZE", 1024)                    # 타일 설정이 바뀌면 타일 결과만 다시 만든다
    assert await run("tile") == "text #3" and await run("whole") == "text #1"


async def test_pages_transcribed_in_another_mode_are_transcribed_again():
    calls = []

    async def read(attachment, _instruction):
        calls.append(attachment.name)
        return f"{attachment.name} read #{len(calls)}"

    root = Attachment(name="drawing.pdf", kind="pdf", mime="application/pdf", text="native")
    on_demand = Attachment(name="drawing.pdf · page 9", kind="image", mime="image/png", data=b"p9", has_data=True, page_number=9)
    photo = Attachment(name="photo.png", kind="image", mime="image/png", data=b"ph", has_data=True, send_to_model=True)
    attachments = [root, ocr_page(1, b"p1"), ocr_page(2, b"p2"), on_demand, photo]

    output, processed = await prepare_visual_ocr_evidence(attachments, read_image=read, cache=OcrCache(), image_mode="whole")
    assert processed and len(calls) == 2 and [item.ocr_variant for item in output[1:3]] == ["whole", "whole"]

    same, processed = await prepare_visual_ocr_evidence(output, read_image=read, cache=OcrCache(), image_mode="whole")
    assert not processed and len(calls) == 2                          # 같은 모드: 저장된 전사를 그대로 쓴다

    progress = []
    switched, processed = await prepare_visual_ocr_evidence(same, read_image=read, cache=OcrCache(), image_mode="tile",
                                                            on_progress=progress.append)
    assert processed and calls[2:] == ["drawing.pdf · page 1", "drawing.pdf · page 2"]     # 전사했던 쪽만 다시
    assert "이미지 처리 방식이 바뀌어 2쪽을 다시 전사" in progress[0]
    evidence = [item for item in switched if item.name == "drawing.pdf · visual OCR"]
    assert len(evidence) == 1 and "read #3" in evidence[0].text and "read #1" not in evidence[0].text
    assert all(item.ocr_variant == config.image_mode_variant("tile") for item in switched[1:3])
    assert on_demand.ocr_variant is None and photo.ocr_variant is None and photo.send_to_model


async def test_pages_transcribed_before_step_5_count_as_whole_mode():
    """Step 5 이전에 전사된 쪽에는 방식 기록이 없다. 증거 문서에 그 쪽이 있으면 전체 모드로 전사된 것이다."""
    calls = []

    async def read(attachment, _instruction):
        calls.append(attachment.name)
        return "new text"

    def legacy():
        page = ocr_page(1, b"p1")
        page.ocr_required = False
        evidence = Attachment(name="drawing.pdf · visual OCR", kind="document", mime="text/plain",
                              text="[VISUAL SOURCE: drawing.pdf · page 1]\nold text")
        return [Attachment(name="drawing.pdf", kind="pdf", mime="application/pdf", text="native"), page, evidence]

    _, processed = await prepare_visual_ocr_evidence(legacy(), read_image=read, cache=OcrCache(), image_mode="whole")
    assert not processed and calls == []
    output, processed = await prepare_visual_ocr_evidence(legacy(), read_image=read, cache=OcrCache(), image_mode="tile")
    assert processed and calls == ["drawing.pdf · page 1"] and "new text" in output[-1].text


# --------------------------------------------------------------------------- /api/chat: 요청마다 모드 선택
def seeing_model(body):
    if is_ocr_call(body):
        return transcribe(request_images(body)[0])
    if is_grounding_call(body):
        return box_reply(request_images(body)[0])
    return "도면 번호는 FA-7731-B입니다."


def evidence_of(client, conversation: str) -> str:
    store = client.app.state.store
    attachments = client.portal.call(store.list_attachments, conversation)
    return next(item.text for item in attachments if item.name.endswith("visual OCR"))


def test_health_reports_the_default_mode_and_tile_settings(client):
    health = client.get("/api/health").json()
    assert health["imageMode"] == "whole"
    assert health["tiling"] == {"tileSize": 1536, "overlap": 0.125, "renderDpi": 200, "minSourceEdge": 2048, "maxTiles": 48}


def test_tile_mode_is_chosen_per_request_and_recorded_on_the_answer(client, mock_llm):
    mock_llm.reset(seeing_model)
    body = chat_body(mock_llm, "도면 번호 알려줘", [upload("scan.pdf", scanned_drawing(), PDF)], imageMode="tile")
    data = client.post("/api/chat", json=body).json()

    ocr = [request for request in mock_llm.requests if is_ocr_call(request)]
    assert len(ocr) == 3 and all(image_count(request) == 1 for request in ocr)
    assert all(max(request_images(request)[0].size) <= config.TILE_SIZE for request in ocr)
    assert all(TILE_OCR_NOTE in all_text(request) for request in ocr)
    (main,) = [request for request in mock_llm.requests if not is_ocr_call(request)]
    assert image_count(main) == 0                                     # 답변 호출에는 여전히 이미지가 없다
    text = all_text(main)
    assert "[VISUAL SOURCE: scan.pdf · page 1]" in text and "[TILED TRANSCRIPTION: this page was read as 2 rows x 3 columns" in text
    assert f"[TILE r1c1]\n{RED_TEXT}\n{GREEN_TEXT}\n[TILE r2c3]\n{BLUE_TEXT}" in text

    assert data["meta"]["imageMode"] == "tile" and data["meta"]["tiling"]["tileSize"] == 1536
    assert data["meta"]["vision"] == {"ocrCalls": 3, "groundingCalls": 0, "answerCalls": 1, "tiledImages": 1,
                                      "tiles": 3, "blankTiles": 3, "ocrLengthStops": 0, "groundingLengthStops": 0}
    assert data["meta"]["elapsedMs"] >= 0
    saved = client.get(f"/api/sessions/{data['conversationId']}").json()["messages"]
    assert saved[0]["meta"] == {} and saved[1]["meta"] == data["meta"]      # 답변과 함께 저장된다


def test_whole_mode_is_the_default_and_behaves_as_before(client, mock_llm):
    mock_llm.reset(seeing_model)
    data = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호 알려줘", [upload("scan.pdf", scanned_drawing(), PDF)])).json()
    (ocr,) = [request for request in mock_llm.requests if is_ocr_call(request)]
    assert request_images(ocr)[0].size == (3072, 2174)                # 쪽 전체 한 장, 3072px로 축소
    assert TILE_OCR_NOTE not in all_text(ocr)
    assert "TILED TRANSCRIPTION" not in all_text(mock_llm.requests[-1])
    assert data["meta"]["imageMode"] == "whole" and "tiling" not in data["meta"]
    assert data["meta"]["vision"] == {"ocrCalls": 1, "groundingCalls": 0, "answerCalls": 1, "tiledImages": 0,
                                      "tiles": 0, "blankTiles": 0, "ocrLengthStops": 0, "groundingLengthStops": 0}


def test_default_mode_comes_from_config_and_unknown_modes_are_rejected(client, mock_llm, monkeypatch):
    mock_llm.reset(seeing_model)
    bad = client.post("/api/chat", json=chat_body(mock_llm, "hi", imageMode="zoom"))
    assert bad.status_code == 400 and "이미지 처리 방식" in bad.json()["error"] and mock_llm.requests == []

    monkeypatch.setattr(config, "DEFAULT_IMAGE_MODE", "tile")         # DOCCHAT_IMAGE_MODE=tile
    data = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호", [upload("scan.pdf", scanned_drawing(), PDF)])).json()
    assert data["meta"]["imageMode"] == "tile" and data["meta"]["vision"]["tiles"] == 3
    assert client.post("/api/chat", json=chat_body(mock_llm, "안녕", imageMode="WHOLE")).json()["meta"]["imageMode"] == "whole"


def test_changing_the_mode_in_a_conversation_transcribes_again(client, mock_llm):
    """모드를 바꿨는데 이전 전사가 그대로 쓰이면 비교가 무의미하다."""
    mock_llm.reset(seeing_model)
    first = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호", [upload("scan.pdf", scanned_drawing(), PDF)])).json()
    conversation = first["conversationId"]
    assert "TILED TRANSCRIPTION" not in evidence_of(client, conversation)

    def follow(text, mode):
        mock_llm.reset(seeing_model)
        body = chat_body(mock_llm, text, conversationId=conversation, imageMode=mode)
        body["messages"] = [{"role": "user", "content": "도면 번호"}, {"role": "assistant", "content": first["text"]},
                            {"role": "user", "content": text}]
        return client.post("/api/chat", json=body).json()

    same = follow("다시", "whole")
    assert [is_ocr_call(request) for request in mock_llm.requests] == [False] and same["meta"]["vision"]["ocrCalls"] == 0

    tiled = follow("타일로 다시", "tile")
    assert sum(is_ocr_call(request) for request in mock_llm.requests) == 3 and tiled["meta"]["imageMode"] == "tile"
    assert "TILED TRANSCRIPTION" in evidence_of(client, conversation)
    assert "TILED TRANSCRIPTION" in all_text(mock_llm.requests[-1])

    back = follow("전체로 다시", "whole")                               # 캐시에 전체 모드 결과가 있어 모델을 다시 부르지 않는다
    assert sum(is_ocr_call(request) for request in mock_llm.requests) == 0 and back["meta"]["vision"]["ocrCalls"] == 0
    assert "TILED TRANSCRIPTION" not in evidence_of(client, conversation)


def locate(body):
    if is_grounding_call(body):
        return box_reply(request_images(body)[0])
    if any(message.get("role") == "tool" for message in body["messages"]):
        return "표시했습니다."
    return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "plan.png", "task": "find the red square"}}]}


def test_bbox_in_tile_mode_is_measured_on_the_stored_original(client, mock_llm, files_dir):
    mock_llm.reset(locate)
    original = encode(marked_image(4000, 2000, [(3000, 1500, 3200, 1700, "red")]))
    body = chat_body(mock_llm, "빨간 사각형 위치를 표시해줘", [upload("plan.png", original, "image/png")], imageMode="tile")
    data = client.post("/api/chat", json=body).json()

    grounding = [request for request in mock_llm.requests if is_grounding_call(request)]
    assert len(grounding) == 1 and request_images(grounding[0])[0].size == (1462, 1096)      # r2c3 한 장
    # 타일 안의 200px 사각형 = 원본 해상도. (전체 모드의 3072px 사본에서는 154px로 줄어 있다)
    assert color_box(request_images(grounding[0])[0], "red") == (462, 596, 662, 796)
    assert request_images(mock_llm.requests[0])[0].size == (3072, 1536)      # 답변 호출에는 전체 사본 한 장

    (artifact,) = data["artifacts"]
    (found,) = artifact["boxes"]
    assert found["x"] == pytest.approx(0.75, abs=0.002) and found["y"] == pytest.approx(0.75, abs=0.002)
    assert found["w"] == pytest.approx(0.05, abs=0.002) and found["h"] == pytest.approx(0.10, abs=0.002)
    assert data["meta"]["vision"] == {"ocrCalls": 0, "groundingCalls": 1, "answerCalls": 2, "tiledImages": 1,
                                      "tiles": 1, "blankTiles": 5, "ocrLengthStops": 0, "groundingLengthStops": 0}
    # 뷰어가 보여 주는 이미지는 사본이고 박스는 분수 좌표라 그대로 맞는다.
    with Image.open(io.BytesIO(client.get(f"/api/attachments/{artifact['attachmentId']}/content").content)) as shown:
        assert shown.size == (3072, 1536)
    assert not (files_dir / data["conversationId"] / "tiles").exists()       # 트레이스를 켜지 않으면 타일을 남기지 않는다


def test_tiles_sent_to_the_model_are_saved_when_tracing_is_on(client, mock_llm, files_dir, monkeypatch):
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    mock_llm.reset(seeing_model)
    body = chat_body(mock_llm, "도면 번호", [upload("scan.pdf", scanned_drawing(), PDF)], imageMode="tile")
    conversation = client.post("/api/chat", json=body).json()["conversationId"]
    folder = files_dir / conversation / "tiles" / "scan.pdf.page 1" / "ocr-1536px-200dpi-o0.125"
    assert sorted(path.name for path in folder.iterdir()) == ["r01c01.png", "r01c02.png", "r02c03.png"]
    sent = {request_images(request)[0].tobytes() for request in mock_llm.requests if is_ocr_call(request)}
    assert {Image.open(path).tobytes() for path in folder.iterdir()} == sent          # 모델이 실제로 본 그 이미지

    # 같은 설정으로 다시 만들어도 덮어쓰거나 복제하지 않는다.
    again = chat_body(mock_llm, "다시", [upload("scan.pdf", scanned_drawing(), PDF)], imageMode="tile", conversationId=conversation)
    client.post("/api/chat", json=again)
    assert sorted(path.name for path in folder.iterdir()) == ["r01c01.png", "r01c02.png", "r02c03.png"]
    # 대화를 지우면 타일도 함께 지워진다.
    client.delete(f"/api/sessions/{conversation}")
    assert not (files_dir / conversation).exists()
