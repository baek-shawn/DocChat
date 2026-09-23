"""Step 3 — §5.3 OCR 오케스트레이션과 증거 선택."""
from __future__ import annotations

import asyncio

from app import config
from app.attachments import Attachment
from app.pipeline.evidence import (attachment_context_for_prompt, attachment_manifest, attachment_root_name, clip,
                                   clip_visual_ocr_coverage, merge_attachment_sets)
from app.pipeline.ocr import (OcrCache, build_ocr_reader, clean_transcription, is_usable_ocr_response,
                              prepare_visual_ocr_evidence)
from app.providers.base import ModelResponse, Provider


def page(number: int, data: bytes, root: str = "drawing.pdf", classification: str = "scanned-raster") -> Attachment:
    return Attachment(name=f"{root} · page {number}", kind="image", mime="image/png", data=data, has_data=True,
                      page_number=number, page_classification=classification, ocr_required=True)


class ScriptedProvider(Provider):
    name = "scripted"

    def __init__(self, replies):
        super().__init__(model="m")
        self.replies, self.calls = list(replies), []

    async def analyze(self, messages, images=None, tools=None, *, temperature=0.2):
        self.calls.append({"messages": messages, "images": images, "tools": tools, "temperature": temperature})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, ModelResponse) else ModelResponse(text=reply)

    async def list_models(self):
        return ["m"]


# --------------------------------------------------------------------------- 오케스트레이션
async def test_ocr_transcribes_each_page_once_in_source_order():
    calls = []

    async def read(attachment, instruction):
        calls.append((attachment.name, instruction))
        await asyncio.sleep(0.03 if attachment.page_number == 1 else 0)   # 1쪽이 늦게 끝나도 순서는 유지돼야 한다
        return "  PART NO.    QTY\n\n  A-001       02\n" if attachment.page_number == 1 else "SECOND PAGE"

    attachments = [Attachment(name="drawing.pdf", kind="pdf", mime="application/pdf", text="native"),
                   page(1, b"page-1"), page(2, b"page-2", classification="vector-outlines")]
    output, processed = await prepare_visual_ocr_evidence(attachments, read_image=read, cache=OcrCache())

    assert processed and len(calls) == 2
    assert "word for word" in calls[0][1] and "No summary" in calls[0][1] and "page 1" in calls[0][1]
    evidence = next(item for item in output if item.name == "drawing.pdf · visual OCR")
    assert evidence.kind == "document" and evidence.mime == "text/plain"
    assert evidence.text.index("page 1]") < evidence.text.index("page 2]")
    assert "[CLASSIFICATION: vector-outlines]" in evidence.text
    # 공백·빈 줄까지 글자 그대로 보존한다.
    assert "  PART NO.    QTY\n\n  A-001       02" in evidence.text
    # 전사가 끝난 페이지는 더 이상 전사 대상도, 메인 요청 이미지도 아니다(바이트는 표시용으로 남는다).
    pages = [item for item in output if item.kind == "image"]
    assert all(not item.ocr_required and not item.send_to_model and item.data for item in pages)


async def test_ocr_never_exceeds_configured_concurrency():
    running = peak = 0

    async def read(_attachment, _instruction):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return "text"

    attachments = [page(number, f"p{number}".encode()) for number in range(1, 8)]
    await prepare_visual_ocr_evidence(attachments, read_image=read, cache=OcrCache())
    assert peak == config.OCR_CONCURRENCY == 2


async def test_identical_image_is_not_transcribed_twice_and_failures_are_not_cached():
    cache, calls = OcrCache(), []

    async def read(attachment, _instruction):
        calls.append(attachment.name)
        return "[OCR FAILED AFTER 3 ATTEMPTS: boom]" if attachment.data == b"bad" else "stable text"

    for _ in range(2):
        await prepare_visual_ocr_evidence([page(1, b"same"), page(2, b"bad")], read_image=read, cache=cache,
                                          cache_namespace="provider:url:model")
    assert calls.count("drawing.pdf · page 1") == 1      # 두 번째 실행은 캐시 적중
    assert calls.count("drawing.pdf · page 2") == 2      # 실패는 캐시하지 않으므로 다시 시도
    # 모델이 바뀌면(네임스페이스가 다르면) 다시 전사한다.
    await prepare_visual_ocr_evidence([page(1, b"same")], read_image=read, cache=cache, cache_namespace="other-model")
    assert calls.count("drawing.pdf · page 1") == 2


def test_cache_is_lru_bounded():
    cache = OcrCache(limit=2)
    for key in "abc":
        cache.remember(key, key)
    assert cache.get("a") is None and cache.get("b") == "b" and len(cache) == 2


async def test_nothing_happens_without_pending_pages():
    attachments = [Attachment(name="photo.png", kind="image", mime="image/png", data=b"x", send_to_model=True)]
    output, processed = await prepare_visual_ocr_evidence(attachments, read_image=None)  # type: ignore[arg-type]
    assert not processed and output is attachments


async def test_pending_page_bytes_are_loaded_on_demand():
    loaded = []

    async def load(attachment):
        loaded.append(attachment.id)
        return b"from-db"

    async def read(attachment, _instruction):
        return f"bytes={attachment.data!r}"

    pending = page(1, b"")
    pending.data, pending.id, pending.has_data = None, 41, True
    output, _ = await prepare_visual_ocr_evidence([pending], read_image=read, load_data=load, cache=OcrCache())
    assert loaded == [41] and "from-db" in output[-1].text


# --------------------------------------------------------------------------- 재시도
def test_refusals_and_commentary_are_not_accepted_as_transcriptions():
    for bad in ("", "   ", "Sorry, I cannot read this.", "I can't help with that", "Here is the transcription:\nABC",
                "The image shows a drawing", "This document contains a table", "죄송합니다. 읽을 수 없습니다.",
                "이 이미지는 도면입니다"):
        assert not is_usable_ocr_response(bad), bad
    for good in ("PART NO. A-001", "[UNCLEAR]", "도면번호 A-1024", "12.5 ±0.1"):
        assert is_usable_ocr_response(good), good


def test_transcription_noise_is_removed_but_content_is_untouched():
    """실제 gemma3 응답에서 관찰된 패턴: 특수 토큰 누출 + 본문 뒤 `[UNCLEAR]` 수백 줄 반복."""
    raw = ("<start_of_image>\n\n  ITEM  PART NO    QTY\n  1     FL-0420    2\n\nDWG NO: FA-7731-B  \n\n"
           + "[UNCLEAR] \n\n" * 300)
    cleaned = clean_transcription(raw)
    assert cleaned == "  ITEM  PART NO    QTY\n  1     FL-0420    2\n\nDWG NO: FA-7731-B\n\n[UNCLEAR]"
    assert len(cleaned) < 100 < len(raw)

    # 표에서 같은 값이 몇 줄 이어지는 것은 정상 내용이다 → 건드리지 않는다.
    table = "QTY\n" + "2\n" * 8 + "END"
    assert clean_transcription(table) == table
    # 같은 줄이 비정상적으로 길게 이어질 때만 줄이고, 줄였다는 사실을 남긴다.
    looped = clean_transcription("HEADER\n" + "NOTE: SEE DETAIL A\n" * 40 + "FOOTER")
    assert looped.count("NOTE: SEE DETAIL A") == 3 and "[REPEATED LINE OMITTED x37" in looped
    assert looped.startswith("HEADER") and looped.endswith("FOOTER")
    # 문서 내용일 수 있는 일반 태그는 특수 토큰으로 오인하지 않는다.
    assert clean_transcription("<html> <b>REV C</b>") == "<html> <b>REV C</b>"
    assert clean_transcription("<think>hmm</think>REV C<|im_end|>") == "REV C"


async def test_reader_retries_bad_answers_then_succeeds_with_literal_settings():
    provider = ScriptedProvider(["Here is the transcription: probably text", RuntimeError("timeout"),
                                 "<think>let me look</think>REV C  QTY 4"])
    text = await build_ocr_reader(provider)(page(1, b"img"), "INSTRUCTION")
    assert text == "REV C  QTY 4" and len(provider.calls) == 3
    first, second = provider.calls[0], provider.calls[1]
    assert first["temperature"] == 0.0 and first["tools"] is None
    assert "transcription engine" in first["messages"][0]["content"]
    assert len(first["images"]) == 1 and first["images"][0].data == b"img"
    assert first["messages"][1]["content"] == "INSTRUCTION"
    assert "previous attempt was not a valid transcription" in second["messages"][1]["content"]


async def test_reader_gives_up_after_configured_attempts():
    provider = ScriptedProvider(["Sorry, no."] * config.OCR_RETRY_COUNT)
    text = await build_ocr_reader(provider)(page(1, b"img"), "INSTRUCTION")
    assert text.startswith("[OCR FAILED AFTER 3 ATTEMPTS") and len(provider.calls) == 3


# --------------------------------------------------------------------------- 증거 선택
def test_root_name_groups_derived_attachments():
    assert attachment_root_name("a.pdf · page 12") == "a.pdf"
    assert attachment_root_name("a.pdf · visual OCR") == "a.pdf"
    assert attachment_root_name("plain.png") == "plain.png"


def test_reupload_replaces_the_whole_group():
    previous = [Attachment(name="a.pdf", kind="pdf", mime="application/pdf", id=1),
                Attachment(name="a.pdf · page 1", kind="image", mime="image/png", id=2),
                Attachment(name="a.pdf · visual OCR", kind="document", mime="text/plain", id=3),
                Attachment(name="b.png", kind="image", mime="image/png", id=4)]
    merged, removed = merge_attachment_sets(previous, [Attachment(name="a.pdf", kind="pdf", mime="application/pdf")])
    assert [item.id for item in removed] == [1, 2, 3]
    assert [(item.name, item.id) for item in merged] == [("b.png", 4), ("a.pdf", None)]


def test_manifest_exposes_exact_names_and_parse_route():
    manifest = attachment_manifest([
        Attachment(name="a.pdf", kind="pdf", mime="application/pdf", text="x" * 50, total_pages=3),
        Attachment(name="a.pdf · page 2", kind="image", mime="image/png", ocr_required=True),
        Attachment(name="a.pdf · visual OCR", kind="document", mime="text/plain", text="y" * 20),
    ])
    assert manifest == '"a.pdf": parts=3, pages=3, images=1, parsedText=70 chars, visualOcr=20 chars, pendingVision=1'
    assert attachment_manifest([]) == "none"


def test_clip_keeps_head_and_tail():
    clipped = clip("A" * 5000 + "B" * 5000, 1000)
    assert clipped.startswith("A") and clipped.endswith("B") and "truncated" in clipped and len(clipped) < 1100
    assert clip("short", 1000) == "short"


def test_ocr_coverage_preview_represents_every_page():
    text = "\n\n".join(f"[VISUAL SOURCE: d.pdf · page {n}]\nPAGE{n}_START " + "z" * 3000 for n in range(1, 11))
    preview = clip_visual_ocr_coverage(text, 4000)
    assert len(preview) <= 4000 and preview.startswith("[COVERAGE PREVIEW: all 10 visual sources")
    assert all(f"PAGE{n}_START" in preview for n in range(1, 11))


def test_text_budget_is_shared_fairly_and_images_do_not_steal_it():
    native = "NATIVE_START\n" + "native exact row\n" * 400 + "NATIVE_END"
    ocr = "[VISUAL SOURCE: drawing.pdf · page 1]\nOCR_START\n" + "ocr exact row\n" * 400 + "OCR_END"
    pages = [Attachment(name=f"drawing.pdf · page {n}", kind="image", mime="image/png", text="Page dimensions only")
             for n in range(1, 21)]
    context = attachment_context_for_prompt("", [
        Attachment(name="drawing.pdf", kind="pdf", mime="application/pdf", text=native), *pages,
        Attachment(name="drawing.pdf · visual OCR", kind="document", mime="text/plain", text=ocr),
    ], 12_000, 2)
    by_name = {item.name: item for item in context.documents}
    assert len(by_name["drawing.pdf"].text) > 4000 and "NATIVE_START" in by_name["drawing.pdf"].text
    assert len(by_name["drawing.pdf · visual OCR"].text) > 4000 and "OCR_START" in by_name["drawing.pdf · visual OCR"].text
    assert context.images == []          # 전사된 PDF 페이지 이미지는 메인 요청에 실리지 않는다


def test_direct_images_are_picked_round_robin_and_named_files_narrow_the_context():
    attachments = [Attachment(name=f"{root}.png", kind="image", mime="image/png", send_to_model=True)
                   for root in ("a", "b", "c")]
    attachments.append(Attachment(name="spec.pdf", kind="pdf", mime="application/pdf", text="SPEC TEXT"))
    context = attachment_context_for_prompt("compare everything", attachments, 10_000, 2)
    assert [item.name for item in context.images] == ["a.png", "b.png"]
    # 질문에서 파일 이름을 짚으면 그 묶음만 본다.
    narrowed = attachment_context_for_prompt("what is in B.PNG?", attachments, 10_000, 12)
    assert [item.name for item in narrowed.images] == ["b.png"] and narrowed.documents == []
