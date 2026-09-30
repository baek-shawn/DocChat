"""수동/종단 테스트용 샘플 문서를 `samples/`에 만든다.

    uv run python scripts/make_samples.py

  native_spec.pdf       텍스트 객체가 충분한 사양서 → 이미지 없이 네이티브 텍스트만 사용
  scanned_drawing.pdf   표제란·부품표를 이미지로 박은 "스캔본" → 비전 전사 대상
  vector_drawing.pdf    선·사각형뿐이고 글자가 거의 없는 도면 → vector-outlines → 비전 전사 대상
  mixed_3pages.pdf      1쪽 네이티브 / 2쪽 스캔 / 3쪽 벡터
  sheet_with_stamp.png  승인 도장·서명·치수선이 있는 이미지 → inspect_visual(bbox) 시험용

타일링(Step 5) 비교용 — A1 @ 200 DPI(6622 x 4677px) 평면도. 작은 글자가 넓게 흩어져 있어 3072px로 줄이면 뭉개진다.
  large_plan.png            업로드 이미지(원본에서 타일을 자른다)
  large_scanned_plan.pdf    같은 그림을 A1 한 쪽에 박은 "스캔본"(타일 영역만 PDF에서 렌더한다)
  large_plan.expect.txt     도면에 실제로 적힌 글자(한 줄에 하나) — 전사 결과와 대조
  large_plan.truth.json     소화기 표시(빨간 원)의 정답 박스와 음성 대조(파란 사각형) — bbox IoU 계산
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"

SPEC_LINES = [
    "DOCUMENT: PUMP SKID ASSEMBLY SPECIFICATION",
    "DRAWING NO: PS-2210-A    REVISION: C    DATE: 2026-03-14",
    "PROJECT: COOLING WATER SYSTEM UPGRADE",
    "",
    "BILL OF MATERIALS",
    "ITEM  PART NO     DESCRIPTION              QTY  MATERIAL",
    "001   BR-1001     BASE FRAME               1    S355JR",
    "002   PM-2040     CENTRIFUGAL PUMP 40-160  2    CAST IRON",
    "003   MT-0075     MOTOR 7.5 KW 4P          2    -",
    "004   VL-0150     GATE VALVE DN150 PN16    4    WCB",
    "005   BT-M16X60   HEX BOLT M16x60          32   8.8 ZN",
    "",
    "NOTES",
    "1. ALL DIMENSIONS IN MILLIMETRES UNLESS OTHERWISE STATED.",
    "2. GENERAL TOLERANCE ISO 2768-mK. SURFACE FINISH Ra 3.2.",
    "3. HYDROSTATIC TEST PRESSURE 24 BAR FOR 30 MINUTES.",
]


def font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def title_block_image(width: int = 1654, height: int = 1169) -> Image.Image:
    """A3 가로 비율의 '스캔된 도면' — 표제란과 부품표가 전부 픽셀이다."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, width - 30, height - 30), outline="black", width=4)
    # 간단한 형상
    draw.rectangle((200, 220, 900, 620), outline="black", width=3)
    draw.ellipse((420, 300, 680, 560), outline="black", width=3)
    draw.line((200, 700, 900, 700), fill="black", width=2)
    draw.line((200, 680, 200, 720), fill="black", width=2)
    draw.line((900, 680, 900, 720), fill="black", width=2)
    draw.text((500, 660), "700", font=font(30), fill="black")
    # 부품표
    rows = [("ITEM", "PART NO", "DESCRIPTION", "QTY"), ("1", "FL-0420", "FLANGE DN100", "2"),
            ("2", "GK-0100", "GASKET DN100", "2"), ("3", "BT-M20X80", "STUD BOLT M20x80", "16")]
    top = 120
    for index, row in enumerate(rows):
        y = top + index * 52
        draw.rectangle((980, y, width - 60, y + 52), outline="black", width=2)
        for x, value in zip((995, 1090, 1260, 1530), row):
            draw.text((x, y + 10), value, font=font(26), fill="black")
    # 표제란
    draw.rectangle((980, height - 300, width - 60, height - 60), outline="black", width=3)
    for offset, line in enumerate(("TITLE: FLANGE ADAPTER ASSEMBLY", "DWG NO: FA-7731-B", "REV: D      SCALE: 1:5",
                                   "DRAWN: H. PARK   CHECKED: S. LEE")):
        draw.text((1000, height - 285 + offset * 54), line, font=font(28), fill="black")
    return image


def stamped_sheet() -> Image.Image:
    image = title_block_image()
    draw = ImageDraw.Draw(image)
    # 승인 도장(빨간 원)과 서명(파란 곡선)
    draw.ellipse((120, 820, 420, 1080), outline=(200, 20, 20), width=8)
    draw.text((165, 900), "APPROVED", font=font(44), fill=(200, 20, 20))
    draw.text((190, 960), "2026-03-14", font=font(30), fill=(200, 20, 20))
    points = [(560, 1010), (600, 940), (640, 1030), (690, 930), (740, 1020), (800, 950), (860, 1000)]
    draw.line(points, fill=(20, 40, 170), width=6, joint="curve")
    draw.text((560, 1045), "Signature", font=font(24), fill=(90, 90, 90))
    return image


PLAN_SIZE = (6622, 4677)       # A1 가로 @ 200 DPI
PLAN_PAGE = (2384, 1684)       # A1 가로(pt)
ROOM_USES = ("OFFICE", "STORAGE", "MEETING", "PANTRY", "SERVER", "LOBBY", "ARCHIVE", "LAB", "COPY", "LOUNGE", "WC", "PLANT")
FINISHES = ("VINYL TILE", "CARPET", "EPOXY", "CERAMIC", "RAISED FLOOR")
RED, BLUE = (215, 30, 30), (30, 70, 215)


def floor_plan() -> tuple[Image.Image, list[str], dict]:
    """(그림, 도면에 적힌 글자 목록, bbox 정답). 방 12개 · 방 이름·면적·치수 · 실내 마감표 · 표제란."""
    width, height = PLAN_SIZE
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    written: list[str] = []
    targets: list[dict] = []
    distractors: list[dict] = []

    def write(position: tuple[int, int], text: str, size: int) -> None:
        draw.text(position, text, font=font(size), fill="black")
        written.append(text)

    draw.rectangle((60, 60, width - 60, height - 60), outline="black", width=6)
    left, top, room_width, room_height = 300, 300, 1450, 1150
    for index in range(12):
        row, col = divmod(index, 4)
        x0, y0 = left + col * room_width, top + row * room_height
        x1, y1 = x0 + room_width, y0 + room_height
        draw.rectangle((x0, y0, x1, y1), outline="black", width=10)                     # 벽
        draw.arc((x0 - 130, y1 - 380, x0 + 130, y1 - 120), 270, 360, fill="black", width=3)   # 문
        write((x0 + 80, y0 + 90), f"ROOM {101 + index} {ROOM_USES[index]}", 28)
        write((x0 + 80, y0 + 140), f"AREA {16 + index}.{(index * 7) % 10} m2", 22)
        draw.line((x0 + 60, y1 - 70, x1 - 60, y1 - 70), fill="black", width=2)             # 치수선
        write((x0 + room_width // 2 - 40, y1 - 110), str(3600 + index * 125), 24)
        # 소화기 표시(빨간 원) = 찾을 대상, 파란 사각형 = 음성 대조
        if index in (0, 5, 6, 11):
            cx, cy = x0 + 1100, y0 + 500
            draw.ellipse((cx - 50, cy - 50, cx + 50, cy + 50), fill=RED)
            targets.append({"label": f"fire extinguisher, room {101 + index}", "bbox": [cx - 50, cy - 50, cx + 50, cy + 50]})
        if index in (2, 9):
            bx, by = x0 + 1050, y0 + 450
            draw.rectangle((bx, by, bx + 100, by + 100), fill=BLUE)
            distractors.append({"label": f"blue square, room {101 + index}", "bbox": [bx, by, bx + 100, by + 100]})

    # 실내 마감표(작은 글자)
    table_left, table_top = 300, 3880
    for line, cells in enumerate([("NO", "ROOM", "FLOOR FINISH", "CEILING HT")] + [
            (str(101 + index), ROOM_USES[index], FINISHES[index % len(FINISHES)], f"{2400 + index * 50} mm")
            for index in range(5)]):
        y = table_top + line * 62
        draw.rectangle((table_left, y, table_left + 2300, y + 62), outline="black", width=2)
        for x, value in zip((20, 240, 760, 1600), cells):
            draw.text((table_left + x, y + 16), value, font=font(24), fill="black")
        written.append("  ".join(cells))

    # 표제란
    block_left, block_top = width - 1900, height - 700
    draw.rectangle((block_left, block_top, width - 120, height - 120), outline="black", width=5)
    for line, text in enumerate(("PROJECT: RIVERSIDE OFFICE BLOCK B", "TITLE: SECOND FLOOR PLAN", "DWG NO: AR-2044-C",
                                 "REV: B   SCALE: 1:50   DATE: 2026-05-08", "DRAWN: J. MOON   CHECKED: Y. SEO")):
        write((block_left + 50, block_top + 50 + line * 100), text, 36)
    truth = {"image": {"width": width, "height": height}, "targets": targets, "distractors": distractors}
    return image, written, truth


def png_of(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def add_native_page(document: pymupdf.Document) -> None:
    page = document.new_page(width=595, height=842)
    for index, line in enumerate(SPEC_LINES):
        page.insert_text((56, 80 + index * 20), line, fontsize=10.5, fontname="cour")


def add_scanned_page(document: pymupdf.Document) -> None:
    page = document.new_page(width=1190, height=842)          # A3 가로
    page.insert_image(page.rect, stream=png_of(title_block_image()))


def add_vector_page(document: pymupdf.Document) -> None:
    page = document.new_page(width=1190, height=842)
    page.draw_rect(pymupdf.Rect(30, 30, 1160, 812), color=(0, 0, 0), width=2)
    for step in range(12):
        page.draw_line((120 + step * 60, 150), (120 + step * 60, 600), color=(0, 0, 0), width=0.8)
        page.draw_line((120, 150 + step * 40), (780, 150 + step * 40), color=(0, 0, 0), width=0.8)
    page.draw_circle((950, 400), 120, color=(0, 0, 0), width=1.5)
    page.insert_text((900, 760), "A-1", fontsize=14)            # 글자가 24자에 한참 못 미친다


def save_pdf(name: str, *builders) -> None:
    document = pymupdf.open()
    for build in builders:
        build(document)
    document.set_metadata({"title": name, "author": "DocChat sample generator"})
    document.save(OUT / name)
    document.close()


def main() -> int:
    OUT.mkdir(exist_ok=True)
    save_pdf("native_spec.pdf", add_native_page)
    save_pdf("scanned_drawing.pdf", add_scanned_page)
    save_pdf("vector_drawing.pdf", add_vector_page)
    save_pdf("mixed_3pages.pdf", add_native_page, add_scanned_page, add_vector_page)
    stamped_sheet().save(OUT / "sheet_with_stamp.png")

    plan, written, truth = floor_plan()
    plan_png = png_of(plan)
    (OUT / "large_plan.png").write_bytes(plan_png)
    document = pymupdf.open()
    page = document.new_page(width=PLAN_PAGE[0], height=PLAN_PAGE[1])
    page.insert_image(page.rect, stream=plan_png)
    document.set_metadata({"title": "large_scanned_plan.pdf", "author": "DocChat sample generator"})
    document.save(OUT / "large_scanned_plan.pdf", deflate=True)     # 압축하지 않으면 90MB라 업로드 한도(64MB)를 넘는다
    document.close()
    (OUT / "large_plan.expect.txt").write_text("\n".join(written) + "\n", encoding="utf-8")
    (OUT / "large_plan.truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    for path in sorted(OUT.iterdir()):
        print(f"{path.name:24s} {path.stat().st_size / 1024:8.1f} KB")
    print(f"\n샘플을 만들었습니다: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
