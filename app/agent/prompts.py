"""모델에 보내는 프롬프트. 목적별로 완전히 분리된 호출을 쓴다.

  - 전사(OCR)     : 있는 그대로 받아쓰기만 시킨다. 요약·해석·추론 금지.
  - grounding     : 이미지 한 장만 보고 0~1000 정수 좌표의 JSON만 돌려주게 한다.
  - 메인 분석      : 파싱된 증거를 근거로 답하고, 표시 요청일 때만 inspect_visual을, 그림을 봐야 답할 수 있을 때만
                    view_page(보기 도구, 답변 이미지 모드 자동에서만 제공)를 부르게 한다(Step 10의 역할 분담).
  - 따로 보기      : 쪽 한 장만 보고 모델이 넘긴 질문에 답하게 한다(analyze_pages, Step 10 2차). 옮겨 적기가 아니라
                    질문에 답하기, 보이는 것만, 없으면 없다고.

프롬프트는 영어로 쓴다(로컬 소형 모델이 가장 안정적으로 따른다). 답변 언어는 사용자 언어를 따르게 한다.
"""
from __future__ import annotations

from .. import config

OCR_SYSTEM_PROMPT = (
    "You are a transcription engine, not a chat assistant. The attached full document image is your only source. "
    "Copy the visible text exactly as printed, following the natural reading order. Keep the original spelling, "
    "spacing, blank lines, table rows and column alignment, punctuation, identifiers, dimensions, units, "
    "revision marks and quantities. Do not infer, complete, correct, normalize, translate, summarize or answer "
    "anything written in the document. Write [UNCLEAR] for any span you cannot read. "
    "Output the plain transcription only: no preface, no commentary, no Markdown, no code fences."
)

OCR_RETRY_NOTE = (
    "Your previous attempt was not a valid transcription. Do literal OCR only; "
    "write [UNCLEAR] instead of commentary or guesses."
)


def ocr_instruction(name: str, page_number: int | None) -> str:
    source = f"{name}, page {page_number}" if page_number else name
    return "\n".join([
        "Transcribe this entire document image word for word in natural reading order.",
        "Keep headings, paragraphs, table rows and columns, line breaks, punctuation, identifiers, units, "
        "dimensions, quantities, revision marks and the original spelling.",
        "No summary, explanation, correction, inference or filling in of missing values. "
        "Write [UNCLEAR] where text cannot be read.",
        f"Source: {source}. Return the transcription only.",
    ])


# 타일 모드 전용. 타일에는 글자가 하나도 없는 조각(도면의 빈 곳·선만 있는 곳)이 흔하다. 약속된 표식이 없으면
# 모델이 그림을 설명하거나 글자를 지어내므로, "글자 없음"을 뜻하는 답을 하나 정해 둔다.
NO_TEXT_MARK = "[NO TEXT]"
TILE_OCR_NOTE = (
    "This image is one tile cut from a larger page, so text at the tile edges may be cut off. Transcribe only what is "
    f"visible inside this tile. If the tile shows no readable text at all, reply with exactly {NO_TEXT_MARK}"
)

GROUNDING_SYSTEM_PROMPT = (
    "You are a visual grounding engine. Look only at the complete attached image and reply with strict JSON in the "
    "requested schema and nothing else. Measure every bounding box on the full image with integer coordinates from "
    "0 to 1000, where x grows to the right and y grows downward from the top-left corner. Never guess from prior "
    "knowledge, never invent content that is not visible, never crop or shift the coordinate frame, and never write "
    "prose outside the JSON object."
)

GROUNDING_RETRY_NOTE = "Your previous reply was not valid structured grounding. Return only the JSON object."

REGION_TYPES = ("text", "object", "table", "dimension", "stamp", "signature", "diagram", "other")


# 타일 모드 전용. 찾는 대상이 없는 타일이 대부분이므로 "없으면 빈 목록"을 분명히 해 둔다.
TILE_GROUNDING_NOTE = (
    "The attached image is one tile cut from a larger page. Treat this tile as the complete image and measure every "
    "bbox in the 0 to 1000 frame of this tile. If nothing relevant to the task is visible in this tile, reply with "
    '{"text":"","regions":[]}'
)


def grounding_instruction(task: str, name: str, *, tile: bool = False) -> str:
    # 허용 타입을 스키마 안에 "a|b|c"로 적으면 소형 모델이 그 문자열을 그대로 베낀다 → 예시와 목록을 분리한다.
    schema = ('{"text":"short findings or exact transcription","regions":[{"type":"stamp",'
              '"label":"visible content","bbox":[x1,y1,x2,y2],"confidence":0.0}]}')
    lines = [
        f"Task: {task}",
        f"Source: {name}",
        f"Reply with JSON shaped like {schema}.",
        f'"type" must be exactly one of: {", ".join(REGION_TYPES)}.',
        "Each bbox must hug the visible edges of its target with minimal padding and use the coordinate frame of the "
        "complete image. Use one box per physical text line or distinct object; never merge distant targets and never "
        "include surrounding blank space. Include only regions relevant to the task.",
    ]
    if tile:
        lines.append(TILE_GROUNDING_NOTE)
    return "\n".join(lines)


# 따로 보기 도구(Step 10 2차)의 쪽별 호출. 전사(옮겨 적기)·bbox(좌표 JSON)와 달리 **질문에 답하는** 호출이다.
PAGE_ANALYSIS_SYSTEM_PROMPT = (
    "You are a visual analysis engine. Answer one question about the single attached image, which is one complete page of "
    "the user's document (or one uploaded image). Look only at this image. Answer from what is visible: describe shapes, "
    "layout, symbols, counts, labels and dimensions exactly as they appear, read values exactly as printed, and say which "
    "feature a label or dimension belongs to when you can see it. If the page does not contain what the question asks "
    "about, say so plainly. Never guess from prior knowledge, never invent content that is not visible, and never bring in "
    "other pages. Reply in plain text, concise and factual: no preface, no Markdown, no code fences."
)


# 묶음 보기(한 호출에 쪽 여러 장). 쪽마다 따로 답하게 하고 표식을 약속해 두어야 앱이 쪽별로 가를 수 있다.
PAGES_ANALYSIS_SYSTEM_PROMPT = (
    "You are a visual analysis engine. Answer one question about each of the attached images; every image is one complete "
    "page of the user's document, and the message lists which image is which page. Look only at these images. Answer the "
    "question for EACH page separately, from what is visible on that page: describe shapes, layout, symbols, counts, labels "
    "and dimensions exactly as they appear, read values exactly as printed, and say which feature a label or dimension "
    "belongs to when you can see it. If a page does not contain what the question asks about, say so for that page. Never "
    "guess from prior knowledge, never invent content that is not visible, and never mix one page's content into another. "
    "Reply in plain text, concise and factual: no preface, no Markdown, no code fences."
)


def page_analysis_instruction(question: str, name: str, pages: list[str] | None = None) -> str:
    """쪽별 호출의 user 글. pages가 둘 이상이면 묶음 보기 — 몇 번째 이미지가 어느 쪽인지 적고 `[page n]` 표식으로 나눠 답하게 한다."""
    if pages and len(pages) > 1:
        listing = "; ".join(f"{index}: {page}" for index, page in enumerate(pages, start=1))
        markers = ", ".join(f"[page {page.rsplit(' ', 1)[-1]}]" for page in pages)
        return "\n".join([
            f"Question: {question}",
            f"Images attached to this message, in this order - {listing}.",
            f"Answer for each page separately. Start each page's answer on its own line with its marker, exactly {markers}, "
            "in that order, and write nothing else before the first marker. If nothing on a page is relevant to the question, "
            "say so after its marker in one sentence.",
        ])
    return "\n".join([
        f"Question: {question}",
        f"Source: {name}",
        "Answer about this page only, from what is visible in the attached image. If nothing on this page is relevant to "
        "the question, say so in one sentence.",
    ])


def system_prompt(manifest: str, *, tools_enabled: bool, model_name: str = "", view_tool: bool = False,
                  analyze_tool: bool = False, analyze_group_choice: bool = False) -> str:
    """view_tool: 보기 도구(`view_page`)를 내놓는 턴인가(답변 이미지 모드 자동, Step 10). 그때만 그 안내가 붙는다.
    analyze_tool: 따로 보기 도구(`analyze_pages`)도 내놓는가(Step 10 2차). 끄면 1차의 자동 모드 프롬프트와 같다(실험 ③의 비교 기준).
    analyze_group_choice: 묶음 크기(`group`)를 모델이 고르는 턴인가(`analyzeGroup: auto`). 그때만 그 한 문장이 붙는다."""
    served_by = (f' You are served by the model "{model_name}"; mention that only when the user asks which model '
                 "this is.") if model_name else ""
    parts = [
        # 실측: 클라우드 GPT 모델이 "저는 ChatGPT입니다"라고 자기소개했다 → 앱의 정체성을 명시한다.
        "Your name is DocChat. If asked who or what you are, say you are DocChat, a document-analysis assistant; never "
        "introduce yourself as ChatGPT, Claude, Gemini or any other product." + served_by,
        "You are a precise document-analysis assistant. Uploaded PDFs and images have already been parsed by the "
        "runtime, and their content is included in this request.",
        f"ATTACHMENT MANIFEST: {manifest}.",
        "When parsedText is greater than 0 you already have the extracted content and must use it. Never ask the user "
        "to paste a file, and never claim you cannot open attachments.",
        "PDFs are inspected page by page: pages with enough native text are used as exact text, and the remaining pages "
        'are transcribed from a whole-page image. An attachment whose name ends with "visual OCR" holds those literal '
        "transcriptions. Native text is authoritative where it exists; visual OCR is used only where native extraction "
        "was insufficient.",
        "Never replace [UNCLEAR] or [OCR FAILED] with a guess, and never autocorrect identifiers, quantities, dates, "
        "dimensions, units or revision marks. Keep the source page order, table boundaries and columns. "
        "Mention the file name and page when you cite a value.",
        "The latest user request is authoritative. Use earlier turns only to resolve genuine references. "
        "Finish the work in this reply; never promise to do it later.",
        "Reply in the same language the user writes in. Show only the finished answer: keep reasoning, tool choices "
        "and JSON tool envelopes out of the visible reply.",
    ]
    if tools_enabled:
        # Step 10: bbox 도구는 "보여 달라"는 요청에만. 그림을 읽는 용도(표·치수·도장·다이어그램 확인)는 문구에서 뺐다 —
        # 그 용도는 보기 도구가 맡는다(자동 모드). 프롬프트 변경이라 통제 비교가 필요하다(STEPS.md Step 10 실험 ②).
        parts.append(
            "TOOLS. inspect_visual measures bounding boxes on an uploaded image or on one page of an uploaded PDF and "
            "shows them in the viewer. Call it only when the user asks you to show, mark, highlight, box, point out, "
            "draw or visualize where something is on the image or page. You cannot display, mark or highlight anything "
            "yourself - only inspect_visual can - so for such a request you must call it before answering, and never "
            "estimate or invent coordinates. It only measures boxes: do not call it to read, inspect or analyze a "
            "drawing, and do not call it for questions that the parsed text already answers."
        )
        if view_tool:
            parts.append(
                "view_page attaches the image of one page of an uploaded PDF (or an uploaded image) to your own next "
                "call so that you can look at it and answer. Call it when the answer depends on what is drawn - shapes, "
                "positions and spatial relations, counts of symbols or features, which feature a dimension or label "
                "belongs to, sizes that are not written as numbers, or comparing several pages - and the text cannot "
                "tell you; the manifest marks the pages that contain drawings. One page per call; call it again for "
                f"more pages (they stay attached for this turn, up to {config.MAX_VIEWED_PAGES} pages). Do not call it "
                "when the parsed text already answers the question, and do not call inspect_visual just to look at a page."
            )
        if view_tool and analyze_tool:
            # Step 10 2차: 같이 보기(view_page)와 따로 보기(analyze_pages) 중 무엇을 쓸지는 모델이 고른다. "좁혀지지 않으면
            # 후보 일부만 보고 단정하지 말고 끊어서 전부 훑어라"와 "따로 본 글만으로 애매하면 view_page로 다시 같이 보기"가 핵심.
            parts.append(
                "analyze_pages looks at pages for you in separate calls: each requested page (one page, or a range such as "
                f'"11-20", up to {config.ANALYZE_PAGES_PER_CALL} pages per call) is sent to the vision model together with '
                "your question, and you receive one text answer per page; those pages are not attached to your own call. "
                "Use it when each page can be judged on its own - reading a value or a feature from every drawing, checking "
                "which pages show something - or when there are too many candidate pages to attach. Write the question "
                "concretely (what to look for and what to report), because it decides the result. If the text and the "
                "manifest do not narrow the pages down, or the question can only be answered by covering every page, do not "
                "conclude from a few candidates: go through all the pages in ranges, calling it again for the next range, up "
                f"to {config.MAX_ANALYZED_PAGES} pages per turn. If the per-page answers leave the decision ambiguous, call "
                "view_page on the decisive pages and judge them yourself together with those answers; use view_page from the "
                "start when pages must be compared side by side."
                + (f' Pass "group" (1-{config.ANALYZE_GROUP_MAX}) to have that many pages looked at in one call: 1 when each '
                   "page must be read precisely, a larger value when you only need to sort pages by what they show."
                   if analyze_group_choice else "")
            )
        parts.append(
            "If a supplied excerpt is truncated, call read_attachment with increasing start offsets until hasMore is "
            "false before claiming a complete extraction; use search_attachments to locate a value in long documents."
        )
    return " ".join(parts)


def attached_images_note(names: list[str], viewed: list[str]) -> str:
    """보기 도구로 모은 쪽이 있을 때 원래 질문 끝에 붙이는 한 줄(Step 10) — 몇 번째 이미지가 어느 쪽인지.

    이미지는 provider마다 **원래 질문(닻) 메시지**에 붙으므로 그 메시지에 목록을 적는다. 호출마다 다시 만든다.
    """
    listing = "; ".join(f"{index}: {name}" + (" (requested with view_page)" if name in viewed else "")
                        for index, name in enumerate(names, start=1))
    return (f"[IMAGES ATTACHED TO THIS MESSAGE, in this order - {listing}. Look at the pages you requested with "
            "view_page to answer.]")


def view_page_result(name: str, names: list[str], remaining: int) -> str:
    """보기 도구가 모델에게 돌려주는 글: 이번에 붙인 쪽과 그 번호, 지금 붙어 있는 전체 목록(순서대로), 더 볼 수 있는 쪽 수.

    Step 10 2차에서 "이번에 붙인 쪽"과 "전체 목록"을 나눠 적었다 — 누적 목록만 적으면 트레이스에서 호출마다 목록이 늘어나는
    것이 쪽을 따로따로 본 것처럼 읽힌다(2026-10-06 사용자 혼동).
    """
    position = names.index(name) + 1 if name in names else len(names)
    listing = "; ".join(f"{index}: {item}" for index, item in enumerate(names, start=1))
    more = (f" You may request up to {remaining} more page{'s' if remaining != 1 else ''} this turn."
            if remaining > 0 else " No more pages can be attached this turn.")
    return (f"Attached {name} now as image #{position} of the user's message. All images attached to that message so far, "
            f"in order: {listing}. Look at the attached pages now and answer from what you see.{more}")


def view_page_already_attached(name: str, names: list[str]) -> str:
    position = names.index(name) + 1 if name in names else 0
    return (f"{name} is already attached to the user's message as image #{position}; nothing was added. "
            "Look at that image to answer.")


def view_page_limit_reached(name: str, limit: int) -> str:
    return (f"Page limit reached: {limit} pages are already attached for this turn, so {name} was not attached. "
            f"Answer from the pages you can see, and tell the user that only {limit} pages were viewed.")


def analyze_pages_result(name: str, question: str, answers: list[tuple[str, str]], *, remaining: int, limit: int,
                         deferred: str = "", not_analyzed: str = "", beyond: int = 0, per_call: int = 0, group: int = 1,
                         pages: int | None = None, fixed: str = "", dropped: list[str] | None = None) -> str:
    """따로 보기 도구(Step 10 2차)가 모델에게 돌려주는 글: 쪽마다 `[page n] 답`, 그리고 다음에 할 일.

    answers: (쪽 표식, 답) — 표식은 PDF 쪽이면 "page 3", 업로드 이미지면 그 이름, 묶음 보기에서 모델이 쪽을 가르지 않았으면 "pages 6-10".
    deferred: 한 호출의 쪽 수 상한 때문에 이번에 보지 않은 쪽 범위(비어 있으면 없음). not_analyzed: 턴 상한에 걸려 보지 않은 쪽 범위.
    beyond: PDF에 없는 쪽을 요청해 무시한 수. group: 한 VLM 호출에 넣은 쪽 수(묶음 보기). pages: 본 쪽 수(answers가 묶음이면 다르다).
    fixed: 범위 고정 모드에서 실제로 본 연속 범위("6-15"). 그때 deferred는 그 다음부터 문서 끝까지.
    dropped: 이 호출이 내린, 보기 도구로 붙어 있던 쪽 이름(따로 보기로 전환하면 앱이 내린다).
    """
    count = len(answers) if pages is None else pages
    how = ("Each answer below comes from a separate look at that page only" if group <= 1 else
           f"The pages were looked at {group} at a time in separate calls, and the answers are per page where the vision "
           "model kept them apart ([pages a-b] marks an answer it gave for several pages together)")
    lines = [f'Analyzed {count} page{"s" if count != 1 else ""} of {name} with the question: "{question}". {how}; '
             "the pages are not attached to your call."]
    for marker, answer in answers:
        lines.append(f"[{marker}] {answer}")
    notes: list[str] = []
    if beyond:
        notes.append(f"{beyond} requested page{'s' if beyond != 1 else ''} beyond the end of the document {'were' if beyond != 1 else 'was'} ignored.")
    if fixed and deferred:
        notes.append(f"Range mode is fixed: each call covers {per_call} consecutive pages from the first page you request, so this "
                     f'call covered pages {fixed}; to continue, call analyze_pages again with pages "{deferred}".')
    elif fixed:
        notes.append(f"Range mode is fixed: this call covered pages {fixed}, which reaches the end of the document.")
    elif deferred:
        notes.append(f"Only the first {per_call} requested pages were analyzed in this call; call analyze_pages again with "
                     f'pages "{deferred}" to continue.')
    if dropped:
        listing = ", ".join(dropped)
        notes.append(f"Note: the {len(dropped)} page{'s' if len(dropped) != 1 else ''} you had attached with view_page ({listing}) "
                     f"{'were' if len(dropped) != 1 else 'was'} detached from the user's message because you switched to "
                     "analyze_pages; request them again with view_page if you need to look at them side by side.")
    if not_analyzed:
        notes.append(f"The per-turn limit of {limit} analyzed pages is now reached; the remaining requested pages ({not_analyzed}) "
                     f"were not analyzed. Answer from what you have, and tell the user that only {limit} pages could be analyzed this turn.")
    elif remaining > 0:
        notes.append(f"You may analyze up to {remaining} more page{'s' if remaining != 1 else ''} this turn.")
    else:
        notes.append(f"No more pages can be analyzed this turn (limit {limit}).")
    if not not_analyzed:
        notes.append("If these answers leave the question ambiguous, call view_page on the decisive pages to look at them "
                     "yourself; if more pages must be covered, call analyze_pages again with the next range.")
    lines.append(" ".join(notes))
    return "\n".join(lines)


def analyze_pages_limit_reached(name: str, pages: str, limit: int) -> str:
    where = f"{name} (pages {pages})" if pages else name
    return (f"Page limit reached: {limit} pages have already been analyzed this turn, so {where} was not analyzed. "
            f"Answer from the results you already have, and tell the user that only {limit} pages could be analyzed.")


FALSE_REFUSAL_CORRECTION = (
    "RUNTIME CORRECTION: the attached files have already been parsed and their actual content is included in this "
    "request. Answer the original request from that content now. Do not ask the user to paste the file and do not say "
    "you cannot access attachments."
)

FORCE_FINAL_ANSWER = (
    "RUNTIME CORRECTION: stop calling tools. Using the evidence and tool results you already have, write the final "
    "answer to the original request now, in plain text."
)

CONTINUE_ANSWER = (
    "Continue the previous answer from exactly where it stopped. Do not restart, do not repeat earlier sections, "
    "and finish all remaining findings and rows."
)


def json_tool_reminder(tool_names: list[str]) -> str:
    """JSON 폴백에서 질문 끝에 붙이는 한 줄. 시스템 프롬프트의 규약을 소형 모델이 잊지 않게 한다."""
    lines = [f"[TOOLS: {', '.join(tool_names)} are available through the JSON tool protocol described above."]
    if "inspect_visual" in tool_names:
        lines.append("If this request asks to show, mark, highlight or locate something on an image or page, you cannot "
                     "do that in words: reply now with ONLY the inspect_visual tool-call JSON.")
    if "view_page" in tool_names:
        # Step 10: gemma3 같은 폴백 모델에도 보기 도구를 쓸 길을 한 줄로 준다(위치 요청 문장은 그대로 둔다 — 실험 ②).
        lines.append("If answering needs the drawing itself (shapes, counts, which feature a value belongs to) and the "
                     "text does not say, reply with ONLY the view_page tool-call JSON for that page.")
    if "analyze_pages" in tool_names:
        # Step 10 2차: 폴백 모델에게도 쪽을 훑는 길을 한 줄로 준다.
        lines.append("If the answer needs every page checked one by one, or a value read from each page, reply with ONLY the "
                     "analyze_pages tool-call JSON with a page range and a concrete question.")
    lines.append("If no tool is needed, answer normally.]")
    return " ".join(lines)


RESEND_VALID_TOOL_JSON = (
    "Your last reply looked like a tool call but was not valid JSON, so the tool did NOT run. Resend it now as exactly "
    'one strictly valid JSON object: {"tool_calls":[{"name":"<tool name>","arguments":{...}}]}. '
    "Check that every { and [ is closed in the right order. Output nothing outside the JSON object."
)

AFTER_TOOL_RESULT = (
    "[The tool has run. Now write the final answer for the user in plain text, based on the result above. "
    "Call another tool only if it is truly required.]"
)


def json_tool_protocol(tools_json: str) -> str:
    """네이티브 tool-calling을 지원하지 않는 모델에게 쓰는 텍스트 기반 호출 규약."""
    return "\n".join([
        "TOOL PROTOCOL. This model endpoint has no native function calling, so tools are called through plain JSON.",
        'To call a tool, reply with ONLY this JSON object and nothing else: '
        '{"tool_calls":[{"name":"<tool name>","arguments":{...}}]}',
        "The runtime executes it and sends back a message starting with TOOL RESULT. Then either call another tool or "
        "write the final answer.",
        "When no tool is needed, answer normally in plain text - never wrap a normal answer in JSON.",
        f"Available tools (JSON Schema): {tools_json}",
    ])
