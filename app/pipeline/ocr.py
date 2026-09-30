"""시각 OCR(§5.3) — needs_vlm 페이지만, 메인 답변 전에, 선택된 VLM 자신에게 전사를 시킨다.

동작은 참고 구현(vectra-web `ocr-orchestrator.mjs` + `server.mjs`의 OCR 재시도)과 같다.
  - 필요한 이미지마다 한 번씩, 동시 OCR_CONCURRENCY개, 원래 순서를 유지한다.
  - 거절/잡담 응답은 OCR_RETRY_COUNT회까지 다시 시도한다.
  - 같은 (모델, 이미지)는 SHA-256 키로 캐시해 재전사하지 않는다. 실패 결과는 캐시하지 않는다.
  - 전사가 끝난 페이지 이미지는 메인 요청에 싣지 않는다(표시용으로만 보관). 전사 텍스트만 컨텍스트에 들어간다.

타일 모드(Step 5)에서는 한 쪽을 겹치는 타일로 나눠 타일마다 전사하고, 읽기 순서대로 이어 붙인다.
  - 겹침 영역 때문에 두 타일에 똑같이 찍힌 줄은 뒤 타일에서 지운다. 문서 내용일 수 있는 반복은 지우지 않는다.
  - 캐시 키와 "이 쪽을 어떤 방식으로 전사했나" 기록에 모드·타일 설정이 들어간다.
    요청 모드가 달라지면 이미 전사된 쪽도 새 방식으로 다시 전사한다(이전 결과가 섞이면 비교가 무의미해진다).

폭주 막기(Step 6-0)
  - 전사 호출은 추론을 끄고(요청마다 고른다) 출력 상한(`config.VISION_MAX_TOKENS`)을 붙여 보낸다.
  - 출력 상한에 닿아 끊긴 호출은 **다시 보내지 않는다**. 다시 보내면 상한만큼의 시간이 또 든다(재시도 3회면 3배).
    실측으로는 같은 타일이 매번 끊기지는 않았다(동시에 나간 호출에 따라 달라진다) — 그래도 다시 보내지 않기로 했다.
    끊긴 본문이 전사의 앞부분이면 "여기서 끊겼다"는 표식과 함께 남기고, 추론 글일 수 있으면 버리고 실패로 남긴다.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from typing import Awaitable, Callable

from .. import config
from ..agent.prompts import OCR_RETRY_NOTE, OCR_SYSTEM_PROMPT, TILE_OCR_NOTE, ocr_instruction
from ..attachments import Attachment
from ..providers.base import ModelResponse, Provider, is_output_length_stop
from .evidence import attachment_root_name, visual_ocr_evidence_name
from .images import ModelImage, TileSource, VisionUsage, assemble_model_images

ReadImage = Callable[[Attachment, str], Awaitable[str]]
LoadData = Callable[[Attachment], Awaitable[bytes | None]]
LoadTileSource = Callable[[Attachment], Awaitable[TileSource | None]]
# (첨부 이름, 목적, 모델에 보낼 타일) — 트레이스를 켰을 때 "모델이 실제로 본 이미지"를 파일로 남긴다.
TileSink = Callable[[str, str, list[ModelImage]], Awaitable[None]]
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
_THINK_OPEN = re.compile(r"<think\b", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
INCOMPLETE_MARK = "[TRANSCRIPTION INCOMPLETE"


class OcrCache:
    """최근 사용 순으로 OCR_CACHE_LIMIT개만 보관하는 프로세스 내 캐시."""

    def __init__(self, limit: int = config.OCR_CACHE_LIMIT):
        self.limit = limit
        self._items: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def key(namespace: str, mime: str, data: bytes, variant: str = "whole") -> str:
        """variant: 이미지 처리 방식과 타일 설정(`config.image_mode_variant`).

        키는 쪽 전체 이미지의 바이트로 만든다. 모드가 달라도 그 바이트는 같으므로, variant를 넣지 않으면
        전체 모드로 전사한 결과가 타일 모드 요청에 그대로 재사용된다.
        """
        digest = hashlib.sha256()
        digest.update(str(namespace).encode("utf-8") + b"\0" + str(mime or "").encode("utf-8") + b"\0")
        digest.update(str(variant or "whole").encode("utf-8") + b"\0")
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


# --------------------------------------------------------------------------- 출력 상한에서 끊긴 응답
def _limit_words() -> str:
    limit = config.vision_max_tokens()
    return f"the output limit of {limit} tokens" if limit else "the model's output limit"


def cut_off_transcription(provider: Provider, response: ModelResponse, *, thinking_disabled: bool) -> str:
    """출력 상한에서 끊긴 응답에서 **전사의 앞부분**만 건진다. 건질 것이 없으면 빈 문자열.

    끊긴 글이 추론일 수 있으면 쓰지 않는다 — 추론 글이 전사로 둔갑해 증거에 들어가면 안 된다.
      - `</think>`가 있으면 그 뒤가 답이다. `<think>`가 열린 채 끝났으면 답은 시작도 못 했다.
      - 태그가 없을 때: 서버가 추론을 따로 떼어 줬거나(reasoning), 추론을 끄고 보냈거나, 추론을 본문에 섞지 않는
        클라우드 provider면 본문은 답이다. 그 밖에는(추론을 켠 로컬 모델, 추론을 떼어 주지 않는 서버) 가릴 수 없다.
    """
    raw = str(response.text or "")
    closed = list(_THINK_CLOSE.finditer(raw))
    if closed:
        raw = raw[closed[-1].end():]
    elif _THINK_OPEN.search(raw):
        return ""
    elif provider.is_local and not response.reasoning and not (thinking_disabled and provider.can_disable_thinking()):
        return ""
    text = clean_transcription(raw)
    return text if is_usable_ocr_response(text) else ""


# --------------------------------------------------------------------------- 타일 전사 병합
_NO_TEXT_REPLY = re.compile(r"^\W*no[ _-]?text\W*$", re.IGNORECASE)
_BRACKET_MARK = re.compile(r"^\[.*\]$")


@dataclass
class TileText:
    """타일 하나의 전사 결과. status: "ok" | "empty"(글자 없음) | "failed"(전사 실패)"""
    row: int
    col: int
    box: tuple[float, float, float, float]
    text: str = ""
    status: str = "ok"


def is_no_text_reply(value: str) -> bool:
    """타일에 글자가 없다는 약속된 답(`[NO TEXT]`)인가. 대소문자·괄호·마침표 차이는 받아 준다."""
    return bool(_NO_TEXT_REPLY.match(str(value or "").strip()))


def _line_key(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _may_be_overlap_duplicate(key: str) -> bool:
    """겹침 중복으로 지워도 되는 줄인가.

    짧은 값과 숫자만 있는 줄(수량 `2`, 치수 `1200`, 날짜)은 표·도면에서 정상적으로 되풀이되므로 절대 지우지 않는다.
    `[UNCLEAR]` 같은 표식도 그대로 둔다 — 타일마다 읽지 못한 자리가 따로 있다는 뜻이다.
    """
    compact = key.replace(" ", "")
    if len(compact) < config.TILE_DEDUPE_MIN_CHARS or _BRACKET_MARK.match(key):
        return False
    return any(character.isalpha() for character in compact)


def _share_area(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return (min(first[2], second[2]) - max(first[0], second[0]) > 1e-9
            and min(first[3], second[3]) - max(first[1], second[1]) > 1e-9)


def drop_overlap_duplicates(parts: list[TileText]) -> list[TileText]:
    """겹침 영역 때문에 이웃한 두 타일에 똑같이 찍힌 줄을 **뒤 타일에서** 지운다(읽기 순서 기준).

    전사 글에는 위치 정보가 없어서 "같은 글자가 두 번 찍힌 것"과 "문서에 원래 두 번 있는 것"을 완전히 가를 수는 없다.
    그래서 지우는 쪽을 좁게 잡는다 — 아래를 모두 만족할 때만 지운다.
      1) 두 타일이 실제로 겹친다(이웃 타일).
      2) 줄이 충분히 길고 글자를 포함한다(`_may_be_overlap_duplicate`).
      3) 그 줄이 두 타일 각각에서 **정확히 한 번씩만** 나온다. 한 타일 안에서 되풀이되는 줄(같은 값이 이어지는 표)은
         문서 내용이므로 건드리지 않는다.
    비교 대상은 앞 타일에 **남아 있는** 줄이다. 앞 타일에서 이미 지워진 줄 때문에 그 다음 타일의 줄까지 연달아
    지워지지 않게 하기 위해서다.
    """
    ordered = sorted(parts, key=lambda part: (part.row, part.col))
    kept: list[tuple[TileText, Counter[str]]] = []
    output: list[TileText] = []
    for part in ordered:
        if part.status != "ok":
            output.append(part)
            continue
        lines = part.text.split("\n")
        counts = Counter(_line_key(line) for line in lines if line.strip())
        neighbours = [seen for earlier, seen in kept if _share_area(earlier.box, part.box)]
        remaining = []
        for line in lines:
            key = _line_key(line)
            if (key and counts[key] == 1 and _may_be_overlap_duplicate(key)
                    and any(seen.get(key) == 1 for seen in neighbours)):
                continue
            remaining.append(line)
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(remaining)).strip("\n")
        cleaned = replace(part, text=text)
        kept.append((cleaned, Counter(_line_key(line) for line in remaining if line.strip())))
        output.append(cleaned)
    return output


def merge_tile_transcriptions(parts: list[TileText], *, rows: int, cols: int, blank: int = 0) -> str:
    """타일별 전사를 읽기 순서(위 → 아래, 왼쪽 → 오른쪽)로 이어 한 쪽의 전사로 만든다."""
    merged = drop_overlap_duplicates(parts)
    with_text = [part for part in merged if part.status == "ok" and part.text.strip()]
    failed = [part for part in merged if part.status == "failed"]
    if not with_text and failed:
        # 한 타일도 읽지 못했다 → 쪽 전체의 전사 실패로 남긴다(전체 모드와 같은 표식).
        return failed[0].text
    without_text = len(merged) - len(with_text) - len(failed) + blank
    header = (f"[TILED TRANSCRIPTION: this page was read as {rows} rows x {cols} columns of overlapping tiles, listed "
              f"left to right and top to bottom. {len(with_text)} tiles contain text, {without_text} contain none"
              + (f", {len(failed)} could not be read" if failed else "")
              + ". Text lying on a tile border may be cut off or appear in two tiles.]")
    blocks = [header]
    for part in merged:
        if part in with_text or part in failed:
            blocks.append(f"[TILE r{part.row}c{part.col}]\n{part.text}")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- 전사 호출
def build_ocr_reader(provider: Provider, *, image_mode: str = "whole", load_tile_source: LoadTileSource | None = None,
                     usage: VisionUsage | None = None, on_progress: Progress | None = None,
                     on_tiles: TileSink | None = None, disable_thinking: bool = False) -> ReadImage:
    """전사 전용 호출(temperature 0, 전용 시스템 프롬프트)을 재시도와 함께 감싼다.

    전체 모드는 쪽마다 한 번, 타일 모드는 타일마다 한 번 호출한다. 어느 쪽이든 동시에 나가는 호출은
    OCR_CONCURRENCY개를 넘지 않는다.
    disable_thinking: 전사 호출의 추론을 끈다(요청마다 고른다). 출력 상한은 `config.VISION_MAX_TOKENS`.
    """
    usage = usage if usage is not None else VisionUsage()
    notify = on_progress or (lambda _message: None)
    limiter = asyncio.Semaphore(config.OCR_CONCURRENCY)

    async def transcribe(image: ModelImage, instruction: str) -> tuple[str, str]:
        """이미지 한 장(쪽 전체 또는 타일)을 전사한다. (상태, 글)을 돌려준다."""
        last_error: object = "unknown error"
        for attempt in range(1, config.OCR_RETRY_COUNT + 1):
            prompt = instruction if attempt == 1 else f"{instruction}\n{OCR_RETRY_NOTE}"
            try:
                async with limiter:
                    usage.ocr_calls += 1
                    response = await provider.analyze(
                        [{"role": "system", "content": OCR_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                        images=[image], temperature=0.0, disable_thinking=disable_thinking,
                        max_tokens=config.vision_max_tokens(),
                    )
                if is_output_length_stop(response.finish_reason):
                    # 출력 상한에 닿았다 → 재시도하지 않는다(다시 보내면 상한만큼의 시간이 또 든다).
                    usage.ocr_length_stops += 1
                    partial = cut_off_transcription(provider, response, thinking_disabled=disable_thinking)
                    if partial:
                        return "ok", (f"{partial}\n{INCOMPLETE_MARK}: the model reached {_limit_words()} here. "
                                      "Text after this point was not transcribed.]")
                    return "failed", (f"{_FAILED_PREFIX}: the model reached {_limit_words()} without producing a "
                                      "transcription. Not retried.]")
                text = clean_transcription(response.text)
                if image.tile is not None and is_no_text_reply(text):
                    return "empty", ""
                if is_usable_ocr_response(text):
                    return "ok", text
                last_error = "empty or non-transcription OCR response"
            except asyncio.CancelledError:
                raise
            except Exception as error:  # 한 페이지의 실패가 전체 답변을 막지 않게 한다.
                last_error = error
        return "failed", f"{_FAILED_PREFIX} AFTER {config.OCR_RETRY_COUNT} ATTEMPTS: {last_error}]"

    async def assemble(attachment: Attachment) -> list[ModelImage]:
        data = attachment.data or b""
        if image_mode == "tile" and load_tile_source is not None:
            try:
                source = await load_tile_source(attachment)
                return await assemble_model_images(attachment.name, attachment.mime, data, purpose="ocr",
                                                   mode=image_mode, source=source)
            except asyncio.CancelledError:
                raise
            except Exception:
                # 타일을 만들지 못해도(원본 PDF를 못 읽는 등) 전사를 포기하지 않는다 → 전체 이미지로 읽는다.
                notify(f"타일을 만들지 못해 전체 이미지로 전사합니다… ({attachment.name})")
        return await assemble_model_images(attachment.name, attachment.mime, data, purpose="ocr")

    async def read(attachment: Attachment, instruction: str) -> str:
        images = await assemble(attachment)
        grid = images[0].grid
        if grid is None:
            _status, text = await transcribe(images[0], instruction)
            return text
        usage.count_images(images)
        if on_tiles is not None:
            await on_tiles(attachment.name, "ocr", images)
        finished = 0

        async def read_tile(image: ModelImage) -> TileText:
            nonlocal finished
            status, text = await transcribe(image, f"{instruction}\n{TILE_OCR_NOTE}")
            finished += 1
            notify(f"타일 전사 중… {attachment.name} ({finished}/{len(images)})")
            row, col = image.tile or (1, 1)
            return TileText(row=row, col=col, box=image.source_box, text=text, status=status)

        parts = await asyncio.gather(*(read_tile(image) for image in images))
        return merge_tile_transcriptions(list(parts), rows=grid.rows, cols=grid.cols, blank=grid.blank)

    return read


def transcribed_with(item: Attachment, attachments: list[Attachment]) -> str | None:
    """이미 전사된 쪽이면 그때 쓴 이미지 처리 방식, 전사한 적이 없는 이미지면 None.

    업로드 이미지와 inspect_visual이 요청 시 렌더한 쪽은 전사 대상이 아니므로 None이다.
    """
    if item.ocr_variant:
        return item.ocr_variant
    # Step 5 이전에 전사된 쪽에는 기록이 없다. 그때는 전체 모드뿐이었다.
    evidence_name = visual_ocr_evidence_name(attachment_root_name(item.name))
    evidence = next((other for other in attachments if other.name == evidence_name), None)
    if evidence is not None and f"[VISUAL SOURCE: {item.name}]" in (evidence.text or ""):
        return "whole"
    return None


async def prepare_visual_ocr_evidence(
    attachments: list[Attachment],
    *,
    read_image: ReadImage,
    load_data: LoadData | None = None,
    on_progress: Progress | None = None,
    cache_namespace: str = "",
    cache: OcrCache | None = None,
    image_mode: str = "whole",
    thinking: bool = False,
) -> tuple[list[Attachment], bool]:
    """(갱신된 첨부 목록, 전사를 수행했는지)를 돌려준다.

    전사 대상: 아직 전사하지 않은 쪽 + **다른 방식으로 전사해 둔 쪽**(이번 요청의 방식으로 다시 전사한다).
    "방식"은 이미지 처리 방식(전체/타일)과 전사 호출의 추론 여부(thinking)다.
    """
    cache = cache if cache is not None else OCR_CACHE
    variant = config.image_mode_variant(image_mode, thinking=thinking)

    def pending(item: Attachment) -> bool:
        if not item.is_image or not (item.data or item.has_data):
            return False
        return item.ocr_required or transcribed_with(item, attachments) not in (None, variant)

    sources = [item for item in attachments if pending(item)]
    if not sources:
        return attachments, False

    total = len(sources)
    notify = on_progress or (lambda _message: None)
    if any(item.ocr_required for item in sources):
        notify(f"텍스트를 읽을 수 없는 {total}쪽을 비전 모델로 전사하는 중…")
    elif all(transcribed_with(item, attachments) in (config.image_mode_variant(image_mode, thinking=True),
                                                      config.image_mode_variant(image_mode, thinking=False))
             for item in sources):
        notify(f"전사 호출의 추론 설정이 바뀌어 {total}쪽을 다시 전사하는 중…")
    else:
        notify(f"이미지 처리 방식이 바뀌어 {total}쪽을 다시 전사하는 중…")
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
            key = OcrCache.key(cache_namespace, source.mime, source.data or b"", variant)
            text = cache.get(key)
            if text is None:
                text = str(await read_image(source, ocr_instruction(source.name, source.page_number)))
                # 실패는 캐시하지 않는다. 타일 하나라도 실패했으면 다음에 다시 시도할 수 있게 쪽 전체를 캐시하지 않는다.
                if _FAILED_PREFIX not in text.upper():
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
        source.ocr_variant = variant

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
