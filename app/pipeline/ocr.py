"""시각 OCR(§5.3) — needs_vlm 페이지만, 메인 답변 전에, 선택된 VLM 자신에게 전사를 시킨다.

동작은 참고 구현(vectra-web `ocr-orchestrator.mjs` + `server.mjs`의 OCR 재시도)과 같다.
  - 필요한 이미지마다 한 번씩, 동시 OCR_CONCURRENCY개, 원래 순서를 유지한다.
  - 거절/잡담 응답은 OCR_RETRY_COUNT회까지 다시 시도한다.
  - 같은 (모델, 이미지)는 SHA-256 키로 캐시해 재전사하지 않는다. 실패 결과는 캐시하지 않는다.
  - 전사가 끝난 페이지 이미지는 메인 요청에 싣지 않는다(표시용으로만 보관). 전사 텍스트만 컨텍스트에 들어간다.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections import OrderedDict
from typing import Awaitable, Callable

from .. import config
from ..agent.prompts import OCR_RETRY_NOTE, OCR_SYSTEM_PROMPT, ocr_instruction
from ..attachments import Attachment
from ..providers.base import Provider
from .evidence import attachment_root_name, visual_ocr_evidence_name
from .images import assemble_model_images

ReadImage = Callable[[Attachment, str], Awaitable[str]]
LoadData = Callable[[Attachment], Awaitable[bytes | None]]
Progress = Callable[[str], None]

_FAILED_PREFIX = "[OCR FAILED"
# 전사가 아니라 사과·설명·요약으로 시작하는 응답을 걸러낸다.
_NOT_A_TRANSCRIPTION = re.compile(
    r"^\s*(?:sorry[,!.]|i (?:cannot|can't|am unable|'m unable)|unable to|as an ai|here (?:is|are)|"
    r"the image (?:shows|contains|depicts)|this (?:image|document|page) (?:shows|contains|is|appears)|"
    r"죄송|이 이미지(?:는|에는)|이미지에는|다음은|제공된 이미지)",
    re.IGNORECASE,
)
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


class OcrCache:
    """최근 사용 순으로 OCR_CACHE_LIMIT개만 보관하는 프로세스 내 캐시."""

    def __init__(self, limit: int = config.OCR_CACHE_LIMIT):
        self.limit = limit
        self._items: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def key(namespace: str, mime: str, data: bytes) -> str:
        digest = hashlib.sha256()
        digest.update(str(namespace).encode("utf-8") + b"\0" + str(mime or "").encode("utf-8") + b"\0")
        digest.update(data)
        return digest.hexdigest()

    def get(self, key: str) -> str | None:
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        return None

    def remember(self, key: str, value: str) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


OCR_CACHE = OcrCache()


def is_usable_ocr_response(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and not _NOT_A_TRANSCRIPTION.match(text)


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK.sub("", str(text or "")).strip()


# 모델이 본문에 흘리는 특수 토큰(실제 gemma3 응답에서 "<start_of_image>"가 관찰됨). 문서 내용일 수 있는
# 일반 태그(<html> 등)는 건드리지 않도록 알려진 토큰만 지운다.
_SPECIAL_TOKENS = re.compile(
    r"<(?:start_of_image|end_of_image|start_of_turn|end_of_turn|bos|eos|pad|image|s|/s)>|"
    r"<\|(?:im_start|im_end|endoftext|vision_start|vision_end|image_pad|begin_of_text|eot_id)\|>",
    re.IGNORECASE,
)
_MAX_IDENTICAL_LINES = 12


def clean_transcription(text: str) -> str:
    """전사 내용은 그대로 두고 **잡음만** 걷어 낸다.

    소형 VLM은 본문을 옳게 받아쓴 뒤 같은 줄을 출력 한도까지 되풀이하는 퇴화 루프에 빠지곤 한다
    (실측: `[UNCLEAR]`가 수백 줄). 그대로 두면 증거 텍스트와 프롬프트 예산이 잡음으로 찬다.
      - `[UNCLEAR]`의 연속은 하나로 줄인다(읽을 수 없는 구간이라는 의미는 그대로다).
      - 그 밖의 같은 줄이 12번 넘게 이어지면 3줄만 남기고, 생략했다는 표식을 남긴다.
    """
    # 들여쓰기는 표의 정렬 단서다 → 첫 줄 앞 공백을 지우지 않도록 전체 strip()은 쓰지 않는다.
    lines = _SPECIAL_TOKENS.sub("", _THINK_BLOCK.sub("", str(text or ""))).replace("\r", "").split("\n")
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        key = line.strip()
        if not key:
            output.append(line)
            index += 1
            continue
        # 빈 줄을 사이에 둔 같은 줄의 연속 구간을 찾는다.
        end, count = index + 1, 1
        probe = end
        while probe < len(lines):
            if not lines[probe].strip():
                probe += 1
                continue
            if lines[probe].strip() != key:
                break
            count += 1
            probe += 1
            end = probe
        if key.upper() == "[UNCLEAR]" and count > 1:
            output.append(line)
        elif count > _MAX_IDENTICAL_LINES:
            output += [line] * 3
            output.append(f"[REPEATED LINE OMITTED x{count - 3}: likely a model repetition loop]")
        else:
            output += lines[index:end]
        index = end
    return re.sub(r"\n{3,}", "\n\n", "\n".join(item.rstrip() for item in output)).strip("\n")


def build_ocr_reader(provider: Provider) -> ReadImage:
    """전사 전용 호출(temperature 0, 전용 시스템 프롬프트)을 재시도와 함께 감싼다."""

    async def read(attachment: Attachment, instruction: str) -> str:
        last_error: object = "unknown error"
        for attempt in range(1, config.OCR_RETRY_COUNT + 1):
            prompt = instruction if attempt == 1 else f"{instruction}\n{OCR_RETRY_NOTE}"
            try:
                pieces = []
                for image in assemble_model_images(attachment.name, attachment.mime, attachment.data or b"", purpose="ocr"):
                    response = await provider.analyze(
                        [{"role": "system", "content": OCR_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                        images=[image], temperature=0.0,
                    )
                    pieces.append(clean_transcription(response.text))
                text = "\n".join(piece for piece in pieces if piece)
                if is_usable_ocr_response(text):
                    return text
                last_error = "empty or non-transcription OCR response"
            except asyncio.CancelledError:
                raise
            except Exception as error:  # 한 페이지의 실패가 전체 답변을 막지 않게 한다.
                last_error = error
        return f"{_FAILED_PREFIX} AFTER {config.OCR_RETRY_COUNT} ATTEMPTS: {last_error}]"

    return read


async def prepare_visual_ocr_evidence(
    attachments: list[Attachment],
    *,
    read_image: ReadImage,
    load_data: LoadData | None = None,
    on_progress: Progress | None = None,
    cache_namespace: str = "",
    cache: OcrCache | None = None,
) -> tuple[list[Attachment], bool]:
    """(갱신된 첨부 목록, 전사를 수행했는지)를 돌려준다."""
    cache = cache if cache is not None else OCR_CACHE
    sources = [item for item in attachments if item.ocr_required and item.is_image and (item.data or item.has_data)]
    if not sources:
        return attachments, False

    total = len(sources)
    notify = on_progress or (lambda _message: None)
    notify(f"텍스트를 읽을 수 없는 {total}쪽을 비전 모델로 전사하는 중…")
    results: list[str] = [""] * total
    cursor = 0
    finished = 0

    async def worker() -> None:
        nonlocal cursor, finished
        while True:
            index = cursor
            cursor += 1
            if index >= total:
                return
            source = sources[index]
            if source.data is None and load_data is not None:
                source.data = await load_data(source)
            key = OcrCache.key(cache_namespace, source.mime, source.data or b"")
            text = cache.get(key)
            if text is None:
                text = str(await read_image(source, ocr_instruction(source.name, source.page_number)))
                if not text.strip().upper().startswith(_FAILED_PREFIX):
                    cache.remember(key, text)
            results[index] = text
            finished += 1
            notify(f"페이지 전사 중… ({finished}/{total})")

    await asyncio.gather(*(worker() for _ in range(min(config.OCR_CONCURRENCY, total))))

    grouped: OrderedDict[str, list[tuple[Attachment, str]]] = OrderedDict()
    for source, text in zip(sources, results):
        grouped.setdefault(attachment_root_name(source.name), []).append((source, text))

    # 전사가 끝난 이미지는 더 이상 전사 대상이 아니다. 바이트는 뷰어/inspect_visual 표시용으로만 남는다.
    for source in sources:
        source.ocr_required = False
        source.send_to_model = False

    output = list(attachments)
    for root, items in grouped.items():
        blocks = []
        for source, text in items:
            lines = [f"[VISUAL SOURCE: {source.name}]"]
            if source.page_classification:
                lines.append(f"[CLASSIFICATION: {source.page_classification}]")
            lines.append(text)
            blocks.append("\n".join(lines))
        body = "\n\n".join(blocks)
        name = visual_ocr_evidence_name(root)
        existing = next((item for item in output if item.name == name), None)
        if existing is not None:
            existing.text, existing.size = body, len(body.encode("utf-8"))
        else:
            output.append(Attachment(name=name, kind="document", mime="text/plain",
                                     size=len(body.encode("utf-8")), text=body))
    return output, True
