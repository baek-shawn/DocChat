"""Step 3 — §5.1 페이지 판별, §5.2 렌더링 캡, 업로드 이미지 정규화."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from app import config
from app.attachments import Attachment, UploadError, sanitize_uploads
from app.pipeline.images import assemble_model_images, cap_scale, capped_size, fit_image_bytes, prepare_uploaded_image
from app.pipeline.pdf import (PdfError, classify_page, count_alnum, render_pdf_for_vision, render_pdf_page_image,
                              run_pdf)
from app.pipeline.preprocess import preprocess_attachments
from conftest import upload
from pdf_factory import build_pdf, jpeg_bytes, png_bytes


# --------------------------------------------------------------------------- §5.1 판별 규칙(순수 함수)
@pytest.mark.parametrize("chars,raster,vector,expected", [
    (0, 0, 0, ("unknown", True)),
    (0, 0, 50, ("vector-outlines", True)),
    (0, 1, 0, ("scanned-raster", True)),
    (23, 0, 9, ("vector-outlines", True)),          # 24자 미만 → 네이티브 사용 불가
    (24, 0, 0, ("native-vector", False)),           # 경계값: 24자면 사용 가능
    (24, 1, 0, ("mixed-needs-vision", True)),       # 래스터가 있는데 120자 미만 → 성긴 오버레이
    (119, 2, 0, ("mixed-needs-vision", True)),
    (120, 2, 0, ("mixed-native", False)),           # 경계값: 120자면 네이티브로 충분
    (5000, 0, 900, ("native-vector", False)),
])
def test_classification_matches_reference_rules(chars, raster, vector, expected):
    assert classify_page(chars, raster, vector) == expected


def test_alnum_count_includes_korean_and_ignores_punctuation():
    assert count_alnum("도면번호 A-1024, 수량: 4개!") == len("도면번호A1024수량4개")


# --------------------------------------------------------------------------- §5.1 실제 PDF
def test_each_page_kind_is_classified_from_real_pdf_content():
    data = build_pdf("native", "scanned", "vector", "mixed_sparse", "mixed_native", "empty")
    inspection = render_pdf_for_vision(data)
    got = [(p.classification, p.needs_vlm) for p in inspection.page_analysis]
    assert got == [
        ("native-vector", False), ("scanned-raster", True), ("vector-outlines", True),
        ("mixed-needs-vision", True), ("mixed-native", False), ("unknown", True),
    ]
    native, scanned, vector = inspection.page_analysis[:3]
    assert native.native_characters >= 120 and native.raster_images == 0
    assert scanned.native_characters == 0 and scanned.raster_images == 1
    assert vector.vector_operations > 0 and vector.raster_images == 0
    assert inspection.total_pages == 6 and inspection.visual_pages == 4 and not inspection.truncated


def test_only_pages_that_need_vision_are_rendered():
    inspection = render_pdf_for_vision(build_pdf("native", "scanned", "native", "vector"))
    assert [page.page_number for page in inspection.pages] == [2, 4]
    assert all(page.mime == "image/png" and page.data.startswith(b"\x89PNG") for page in inspection.pages)
    # 네이티브 텍스트는 페이지 표식과 함께 그대로 보존된다.
    assert "[PAGE 1]" in inspection.native_text and "DRAWING NO A-1024" in inspection.native_text
    assert "[PAGE 2]" not in inspection.native_text


def test_native_pdf_produces_no_images_at_all():
    inspection = render_pdf_for_vision(build_pdf("native", "mixed_native"))
    assert inspection.pages == [] and inspection.visual_pages == 0


def test_inspection_without_rendering_still_counts_visual_pages():
    inspection = render_pdf_for_vision(build_pdf("scanned", "vector"), render_images=False)
    assert inspection.pages == [] and inspection.visual_pages == 2


def test_page_limit_truncates_and_reports_it():
    inspection = render_pdf_for_vision(build_pdf(*["scanned"] * 5), max_pages=2)
    assert inspection.processed_pages == 2 and inspection.total_pages == 5 and inspection.truncated
    assert len(inspection.pages) == 2


def test_page_limit_env_is_clamped(monkeypatch):
    monkeypatch.setenv("DOCCHAT_MAX_PDF_VISUAL_PAGES", "5000")
    assert config.pdf_visual_page_limit() == config.MAX_PDF_VISUAL_PAGES
    monkeypatch.setenv("DOCCHAT_MAX_PDF_VISUAL_PAGES", "garbage")
    assert config.pdf_visual_page_limit() == config.DEFAULT_PDF_VISUAL_PAGES


def test_thresholds_are_tunable_for_cad_pdfs(monkeypatch):
    """CAD PDF용: 임계값을 낮추면 글자가 적은 벡터 페이지도 네이티브로 통과시킬 수 있다."""
    monkeypatch.setattr(config, "NATIVE_MIN_CHARS", 5)
    assert classify_page(10, 0, 100) == ("native-vector", False)


def test_invalid_and_out_of_range_pdf_errors_are_user_facing():
    with pytest.raises(PdfError):
        render_pdf_for_vision(b"%PDF-1.4 definitely broken")
    with pytest.raises(PdfError, match="범위"):
        render_pdf_page_image(build_pdf("native"), page_number=9)


# --------------------------------------------------------------------------- §5.2 렌더링 캡
def test_cap_scale_respects_both_edge_and_pixel_limits():
    assert cap_scale(1000, 800) == 1.0
    width, height, scale = capped_size(9362, 6622)          # A0 @ 200 DPI
    assert scale < 1 and max(width, height) <= 3072 and width * height <= 8_000_000
    assert abs(width / height - 9362 / 6622) < 0.01          # 비율 유지
    width, height, _ = capped_size(3000, 3000)               # 변은 한도 이내지만 픽셀 수가 초과
    assert width * height <= 8_000_000 and width == height


def test_large_page_render_is_capped_and_any_page_renders_on_demand():
    a0 = build_pdf("vector", page_size=(3370, 2384))          # A0 가로(pt)
    page = render_pdf_for_vision(a0).pages[0]
    assert max(page.width, page.height) <= 3072 and page.width * page.height <= 8_000_000
    assert abs(page.width / page.height - 3370 / 2384) < 0.01
    with Image.open(io.BytesIO(page.data)) as image:
        assert image.size == (page.width, page.height)

    # 네이티브 텍스트 페이지는 미리 렌더하지 않지만, inspect_visual이 요청하면 그릴 수 있어야 한다.
    native = render_pdf_page_image(build_pdf("native"), page_number=1)
    assert native.page_classification == "native-vector" and native.width == round(595 * 200 / 72)


async def test_pdf_work_runs_on_the_single_worker_thread():
    names = set()

    def where() -> str:
        import threading
        return threading.current_thread().name

    for _ in range(4):
        names.add(await run_pdf(where))
    assert len(names) == 1 and next(iter(names)).startswith("pdf-worker")


# --------------------------------------------------------------------------- §5.4 업로드 이미지
def test_small_png_and_jpeg_pass_through_untouched():
    for data, mime in ((png_bytes(), "image/png"), (jpeg_bytes(), "image/jpeg")):
        prepared = prepare_uploaded_image(data, mime)
        assert prepared.data == data and prepared.mime == mime and not prepared.resized


def test_exif_rotated_photo_is_straightened():
    """휴대폰으로 찍은 도면 사진: EXIF 방향 6(시계 방향 90°)이면 가로·세로가 바뀌어야 한다."""
    image = Image.new("RGB", (400, 100), (200, 200, 200))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    prepared = prepare_uploaded_image(buffer.getvalue(), "image/jpeg")
    assert (prepared.width, prepared.height) == (100, 400)


def test_oversized_image_is_downscaled_within_limits():
    prepared = prepare_uploaded_image(png_bytes(6000, 3000), "image/png")
    assert prepared.resized and (prepared.source_width, prepared.source_height) == (6000, 3000)
    assert max(prepared.width, prepared.height) <= 3072 and prepared.width * prepared.height <= 8_000_000
    assert prepared.mime == "image/png"


def test_transparent_and_exotic_formats_are_normalized():
    flattened = prepare_uploaded_image(png_bytes(mode="RGBA"), "image/png")
    with Image.open(io.BytesIO(flattened.data)) as image:
        assert image.mode == "RGB" and image.getpixel((300, 190)) == (255, 255, 255)   # 투명 → 흰 배경
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buffer, format="BMP")
    assert prepare_uploaded_image(buffer.getvalue(), "image/bmp").mime == "image/png"
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buffer, format="WEBP")
    assert prepare_uploaded_image(buffer.getvalue(), "image/webp").mime == "image/jpeg"


def test_fit_image_bytes_shrinks_under_api_limit():
    import os
    noisy = Image.frombytes("RGB", (900, 900), os.urandom(900 * 900 * 3))
    buffer = io.BytesIO()
    noisy.save(buffer, format="PNG")
    data, mime = fit_image_bytes(buffer.getvalue(), "image/png", 400_000)
    assert len(data) <= 400_000 and mime == "image/jpeg"
    assert fit_image_bytes(b"tiny", "image/png", 400_000) == (b"tiny", "image/png")


async def test_assemble_model_images_is_the_tiling_seam():
    """전체 모드는 Step 4까지와 같다: 받은 이미지 한 장, 전체 영역. (타일 모드는 test_tiling.py)"""
    images = await assemble_model_images("a.png", "image/png", b"x", purpose="ocr")
    assert len(images) == 1 and images[0].source_box == (0.0, 0.0, 1.0, 1.0)
    assert images[0].data == b"x" and images[0].tile is None and images[0].grid is None


# --------------------------------------------------------------------------- 업로드 정제 · 전처리
def test_only_pdf_and_images_are_accepted():
    with pytest.raises(UploadError, match="PDF와 이미지"):
        sanitize_uploads([upload("notes.docx", b"PK..", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")])
    with pytest.raises(UploadError, match="올바른 PDF"):
        sanitize_uploads([upload("fake.pdf", b"not a pdf", "application/pdf")])
    with pytest.raises(UploadError, match="최대"):
        sanitize_uploads([upload(f"{i}.png", png_bytes(), "image/png") for i in range(13)])


def test_upload_names_are_made_safe_and_unique():
    items = sanitize_uploads([
        upload("C:\\docs\\plan · page 1.png", png_bytes(), ""),     # 경로 제거 + 예약 구분자 치환 + MIME 추론
        upload("plan - page 1.png", png_bytes(), "image/png"),
    ])
    assert [item.name for item in items] == ["plan - page 1.png", "plan - page 1 (2).png"]
    assert items[0].mime == "image/png"


async def test_preprocess_expands_pdf_into_root_and_page_images():
    uploads = sanitize_uploads([upload("drawing.pdf", build_pdf("native", "scanned", "vector"), "application/pdf"),
                                upload("photo.png", png_bytes(), "image/png")])
    parts = await preprocess_attachments(uploads)
    assert [item.name for item in parts] == ["drawing.pdf", "drawing.pdf · page 2", "drawing.pdf · page 3", "photo.png"]
    root, page2, _page3, photo = parts
    assert root.kind == "pdf" and root.visual_pages == 2 and root.total_pages == 3
    assert "[PAGE ANALYSIS]" in root.text and "Page 2: scanned-raster" in root.text and "vision OCR=required" in root.text
    assert "DRAWING NO A-1024" in root.text
    # PDF 페이지 이미지는 전사 대상이고 메인 요청에는 실리지 않는다.
    assert page2.ocr_required and not page2.send_to_model and page2.page_number == 2
    # 일반 이미지는 전사 없이 그대로 VLM으로 간다(§5.4).
    assert photo.send_to_model and not photo.ocr_required


async def test_preprocess_can_skip_visual_assets():
    uploads = sanitize_uploads([upload("scan.pdf", build_pdf("scanned"), "application/pdf")])
    parts = await preprocess_attachments(uploads, include_visual_assets=False)
    assert [item.name for item in parts] == ["scan.pdf"] and parts[0].visual_pages == 1


def test_public_view_never_leaks_bytes():
    public = Attachment(name="a.png", mime="image/png", kind="image", data=b"secret-bytes", id=7, width=10, height=5).to_public()
    assert public["url"] == "/api/attachments/7/content" and public["hasData"] is True
    assert "data" not in public and "base64" not in public
