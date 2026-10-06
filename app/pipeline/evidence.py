"""첨부 묶음(root) 관리와 프롬프트용 증거 선택.

동작은 참고 구현(vectra-web `document-pipeline/evidence.mjs`)과 같다.
  - 한 번의 업로드(root)와 거기서 파생된 페이지 이미지·시각 OCR 문서를 한 묶음으로 다룬다.
  - 프롬프트 예산 안에서 문서별로 텍스트를 공평하게 나누고, 넘치면 앞·뒤를 남기고 자른다.
  - 시각 OCR 문서는 모든 페이지가 조금씩이라도 보이도록 "커버리지 미리보기"로 자른다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .. import config
from ..attachments import Attachment

CHILD_SEPARATOR = " · "
VISUAL_OCR_SUFFIX = "visual OCR"
_CHILD_PATTERN = re.compile(r" · (?:page \d+|visual OCR)", re.IGNORECASE)
_VISUAL_SOURCE_LINE = re.compile(r"^\[VISUAL SOURCE: .+\]$", re.MULTILINE)
# 쪽별 판별 값이 메타데이터에 없는 옛 PDF 첨부(Step 10 이전에 올린 것)는 본문의 [PAGE ANALYSIS] 줄에서 읽는다.
_PAGE_ANALYSIS_LINE = re.compile(
    r"^Page (?P<page>\d+): (?P<classification>[\w-]+); native characters=(?P<chars>\d+); raster images=(?P<raster>\d+); "
    r"vector operations=(?P<vector>\d+); vision OCR=(?P<vlm>required|skipped)$", re.MULTILINE)


def attachment_root_name(name: str) -> str:
    return _CHILD_PATTERN.split(str(name or ""), maxsplit=1)[0]


def page_image_name(root: str, page_number: int) -> str:
    return f"{root}{CHILD_SEPARATOR}page {page_number}"


def visual_ocr_evidence_name(root: str) -> str:
    return f"{root}{CHILD_SEPARATOR}{VISUAL_OCR_SUFFIX}"


def is_visual_ocr(attachment: Attachment) -> bool:
    return attachment.name.lower().endswith(VISUAL_OCR_SUFFIX.lower())


def merge_attachment_sets(previous: list[Attachment], incoming: list[Attachment]) -> tuple[list[Attachment], list[Attachment]]:
    """같은 이름으로 다시 올린 문서는 파생 첨부까지 통째로 교체한다. (병합 결과, 밀려난 항목)을 돌려준다."""
    incoming_roots = {attachment_root_name(item.name) for item in incoming}
    kept = [item for item in previous if attachment_root_name(item.name) not in incoming_roots]
    removed = [item for item in previous if attachment_root_name(item.name) in incoming_roots]
    return kept + incoming, removed


def page_analysis_of(pdf: Attachment) -> list[dict]:
    """PDF 첨부의 쪽별 판별 값. 메타데이터(`page_analysis`)가 있으면 그것, 없으면(옛 첨부) 본문의 [PAGE ANALYSIS] 줄.

    옛 첨부에는 래스터 면적이 없어 래스터 개수로 대신한다(로고만 있는 쪽도 그림으로 보일 수 있다 — 다시 올리면 정확해진다).
    """
    if pdf.page_analysis:
        return [item for item in pdf.page_analysis if isinstance(item.get("page"), int)]
    pages = []
    for match in _PAGE_ANALYSIS_LINE.finditer(pdf.text or ""):
        raster = int(match["raster"])
        pages.append({"page": int(match["page"]), "classification": match["classification"], "chars": int(match["chars"]),
                      "raster": raster, "rasterArea": 1.0 if raster else 0.0, "vector": int(match["vector"]),
                      "vlm": match["vlm"] == "required"})
    return pages


def drawing_pages(pdf: Attachment) -> list[dict]:
    """그림(래스터 그림·벡터 도면)이 있다고 판정한 쪽들(Step 10). 판정 기준은 `config.is_drawing_page`."""
    return [page for page in page_analysis_of(pdf)
            if config.is_drawing_page(raster_area=float(page.get("rasterArea") or 0.0),
                                      vector_operations=int(page.get("vector") or 0),
                                      classification=str(page.get("classification") or ""))]


def page_ranges(numbers: list[int]) -> str:
    """[1, 2, 3, 5, 7, 8] → "1-3, 5, 7-8" (긴 PDF의 쪽 목록을 짧게)."""
    ordered = sorted(set(numbers))
    spans: list[list[int]] = []
    for number in ordered:
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    return ", ".join(str(start) if start == end else f"{start}-{end}" for start, end in spans)


def drawing_cue(pdf: Attachment) -> str:
    """매니페스트에 덧붙이는 판단 재료(Step 10): 어느 쪽에 그림이 있고 그 쪽의 글이 무엇을 담는지.

    모델이 "글로 충분한가"를 추측하지 않게 한다 — 전사한 쪽의 글은 보이는 라벨을 옮겨 적은 것일 뿐이고, 네이티브 글이 있는
    쪽도 그림 자체(형상·배치·개수·어느 대상의 치수인지)는 글에 없다. 그림이 없으면 빈 문자열.
    """
    pages = drawing_pages(pdf)
    if not pages:
        return ""
    transcribed = [page["page"] for page in pages if page.get("vlm")]
    native = [page["page"] for page in pages if not page.get("vlm")]
    parts = []
    if transcribed:
        parts.append(f"{page_ranges(transcribed)} (text = transcription of the visible labels only)")
    if native:
        parts.append(f"{page_ranges(native)} (native text beside the drawing)")
    return (f"drawings on pages {'; '.join(parts)} - the shapes, their positions and counts, and which label or "
            "dimension belongs to which feature are NOT in the text; call view_page to see such a page")


def attachment_manifest(attachments: list[Attachment], *, drawing_cues: bool = False) -> str:
    """시스템 프롬프트에 넣는 한 줄 요약. 모델이 정확한 첨부 이름을 알 수 있게 한다.

    drawing_cues: 보기 도구를 내놓는 턴(답변 이미지 모드 자동, Step 10)에만 True — PDF마다 그림이 있는 쪽을 덧붙인다.
    다른 모드의 매니페스트는 Step 8까지와 같다(실험 ①의 비교 기준).
    """
    if not attachments:
        return "none"
    groups: dict[str, dict] = {}
    for item in attachments:
        group = groups.setdefault(attachment_root_name(item.name), {
            "parts": 0, "images": 0, "text": 0, "visual_ocr": 0, "pending": 0, "pages": 0, "cue": "",
        })
        group["parts"] += 1
        if item.is_image:
            group["images"] += 1
        group["text"] += len(item.text or "")
        if is_visual_ocr(item):
            group["visual_ocr"] += len(item.text or "")
        if item.ocr_required:
            group["pending"] += 1
        if item.kind == "pdf":
            group["pages"] = item.total_pages
            if drawing_cues:
                group["cue"] = drawing_cue(item)
    lines = []
    for name, group in groups.items():
        pages = f", pages={group['pages']}" if group["pages"] else ""
        cue = f"; {group['cue']}" if group["cue"] else ""
        lines.append(
            f'"{name}": parts={group["parts"]}{pages}, images={group["images"]}, parsedText={group["text"]} chars, '
            f'visualOcr={group["visual_ocr"]} chars, pendingVision={group["pending"]}{cue}'
        )
    return "; ".join(lines)


def clip(text: str, max_chars: int) -> str:
    """앞·뒤를 남기고 가운데를 잘라 낸다."""
    if len(text) <= max_chars:
        return text
    half = max(1, (max_chars - 80) // 2)
    return f"{text[:half]}\n\n...[truncated {len(text) - half * 2} chars]...\n\n{text[-half:]}"


def clip_visual_ocr_coverage(text: str, max_chars: int) -> str:
    """OCR한 모든 페이지에 프롬프트 공간을 조금씩 배정한다. 전체 텍스트는 read_attachment 도구로 읽을 수 있다."""
    if len(text) <= max_chars:
        return text
    matches = list(_VISUAL_SOURCE_LINE.finditer(text))
    if not matches:
        return clip(text, max_chars)
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[match.start():end].strip())
    block_budget = max(160, (max_chars - 150) // len(blocks))
    header = (f"[COVERAGE PREVIEW: all {len(blocks)} visual sources are represented; "
              "call read_attachment for the complete text.]")
    return (header + "\n\n" + "\n\n".join(clip(block, block_budget) for block in blocks))[:max_chars]


@dataclass
class PromptContext:
    documents: list[Attachment]   # 텍스트가 예산에 맞게 잘린 복사본
    # 답변 호출에 이미지로 실을 첨부(바이트는 호출부에서 로드). 아직 렌더하지 않은 PDF 쪽은 자리표시(`pending_page_image`)다.
    images: list[Attachment]
    image_candidates: int = 0     # 모드상 실을 수 있었던 이미지 수. 상한 때문에 뺀 수 = image_candidates - len(images)


def pending_page_image(root: str, page_number: int) -> Attachment:
    """아직 렌더하지 않은 PDF 쪽의 자리표시 — id도 바이트도 없다. 호출부(chat_service)가 렌더해 채운다."""
    return Attachment(name=page_image_name(root, page_number), mime="image/png", kind="image", page_number=page_number)


def is_pending_page_image(item: Attachment) -> bool:
    return item.is_image and item.id is None and item.data is None and not item.has_data and bool(item.page_number)


def _answer_image_groups(candidates: list[Attachment], answer_images: str) -> dict[str, list[Attachment]]:
    """답변 이미지 모드(Step 8)에 따라 실을 수 있는 이미지를 업로드 묶음별로 모은다.

    off: 없음 · uploads: 업로드 이미지만(§5.4) · whole: 업로드 이미지 + PDF의 모든 쪽(렌더하지 않은 쪽은 자리표시)
    · auto(Step 10): 처음에는 uploads와 같다 — PDF 쪽은 답변 모델이 보기 도구로 요청할 때 루프가 더한다.
    """
    groups: dict[str, list[Attachment]] = {}
    if answer_images == "off":
        return groups
    for item in candidates:
        if item.is_image and item.send_to_model:      # 업로드 이미지
            groups.setdefault(attachment_root_name(item.name), []).append(item)
    if answer_images != "whole":
        return groups
    for pdf in candidates:
        if not pdf.is_pdf or pdf.kind != "pdf":
            continue
        rendered = {item.page_number: item for item in candidates
                    if item.is_image and item.page_number and attachment_root_name(item.name) == pdf.name
                    and (item.has_data or item.data)}
        total = pdf.total_pages or max(rendered, default=0)
        groups.setdefault(pdf.name, []).extend(
            rendered.get(number) or pending_page_image(pdf.name, number) for number in range(1, total + 1))
    return groups


def attachment_context_for_prompt(prompt: str, attachments: list[Attachment], total_text_chars: int,
                                  max_images: int, *, answer_images: str = "uploads") -> PromptContext:
    if not attachments:
        return PromptContext([], [])
    lower = str(prompt or "").lower()
    named_roots = {attachment_root_name(item.name) for item in attachments if item.name.lower() in lower}
    candidates = [item for item in attachments if attachment_root_name(item.name) in named_roots] if named_roots else attachments

    document_candidates = [item for item in candidates if not item.is_image]
    text_files = [item for item in document_candidates if item.text]
    per_file = max(1000, total_text_chars // max(1, len(text_files)))
    documents = []
    for item in document_candidates:
        text = item.text or ""
        clipped = clip_visual_ocr_coverage(text, per_file) if is_visual_ocr(item) else clip(text, per_file)
        documents.append(replace(item, text=clipped, data=None, source_data=None))

    # 문서별로 한 장씩 돌아가며 뽑아, 긴 문서 하나가 다른 업로드를 밀어내지 못하게 한다(쪽은 쪽 순서).
    groups = _answer_image_groups(candidates, answer_images)
    images: list[Attachment] = []
    round_index = 0
    while len(images) < max_images:
        added = False
        for group in groups.values():
            if round_index < len(group):
                images.append(group[round_index])
                added = True
                if len(images) >= max_images:
                    break
        if not added:
            break
        round_index += 1
    return PromptContext(documents, images, sum(len(group) for group in groups.values()))
