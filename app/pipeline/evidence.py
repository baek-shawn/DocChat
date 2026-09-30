"""첨부 묶음(root) 관리와 프롬프트용 증거 선택.

동작은 참고 구현(vectra-web `document-pipeline/evidence.mjs`)과 같다.
  - 한 번의 업로드(root)와 거기서 파생된 페이지 이미지·시각 OCR 문서를 한 묶음으로 다룬다.
  - 프롬프트 예산 안에서 문서별로 텍스트를 공평하게 나누고, 넘치면 앞·뒤를 남기고 자른다.
  - 시각 OCR 문서는 모든 페이지가 조금씩이라도 보이도록 "커버리지 미리보기"로 자른다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ..attachments import Attachment

CHILD_SEPARATOR = " · "
VISUAL_OCR_SUFFIX = "visual OCR"
_CHILD_PATTERN = re.compile(r" · (?:page \d+|visual OCR)", re.IGNORECASE)
_VISUAL_SOURCE_LINE = re.compile(r"^\[VISUAL SOURCE: .+\]$", re.MULTILINE)


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


def attachment_manifest(attachments: list[Attachment]) -> str:
    """시스템 프롬프트에 넣는 한 줄 요약. 모델이 정확한 첨부 이름을 알 수 있게 한다."""
    if not attachments:
        return "none"
    groups: dict[str, dict[str, int]] = {}
    for item in attachments:
        group = groups.setdefault(attachment_root_name(item.name), {
            "parts": 0, "images": 0, "text": 0, "visual_ocr": 0, "pending": 0, "pages": 0,
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
    lines = []
    for name, group in groups.items():
        pages = f", pages={group['pages']}" if group["pages"] else ""
        lines.append(
            f'"{name}": parts={group["parts"]}{pages}, images={group["images"]}, parsedText={group["text"]} chars, '
            f'visualOcr={group["visual_ocr"]} chars, pendingVision={group["pending"]}'
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
    images: list[Attachment]      # 메인 요청에 이미지로 직접 실을 첨부(바이트는 호출부에서 로드)


def attachment_context_for_prompt(prompt: str, attachments: list[Attachment], total_text_chars: int,
                                  max_images: int) -> PromptContext:
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

    # 문서별로 한 장씩 돌아가며 뽑아, 긴 문서 하나가 다른 업로드를 밀어내지 못하게 한다.
    groups: dict[str, list[Attachment]] = {}
    for item in candidates:
        if item.is_image and item.send_to_model:
            groups.setdefault(attachment_root_name(item.name), []).append(item)
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
    return PromptContext(documents, images)
