"""첨부 파일 저장소 — `data/files/{대화ID}/` 아래에 바이트를 파일로 둔다(Step 5-0).

  - DB에는 이 폴더 기준 **상대 경로**만 들어간다(`{대화ID}/scan.pdf`).
  - 읽고 쓸 때마다 경로를 폴더 안으로 정규화해 검사한다. DB 값이 조작돼도(`..`, 절대 경로, 드라이브 문자)
    폴더 밖의 파일은 읽지도 쓰지도 지우지도 않는다.
  - 파일은 한 번 쓰면 덮어쓰지 않는다. 같은 이름이 있으면 " (2)"를 붙여 새 파일로 쓴다.
  - 지우는 것은 대화를 삭제할 때 그 대화 폴더뿐이다. 실패하면 로그만 남긴다.

여기 함수들은 동기 함수다(디스크 I/O). 이벤트 루프에서는 `asyncio.to_thread`로 부른다.
"""
from __future__ import annotations

import logging
import re
import shutil
import unicodedata
from pathlib import Path

logger = logging.getLogger("docchat.storage")

_FOLDER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_INVALID_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Windows가 파일 이름으로 받지 않는 장치 이름
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}
_MIME_EXTENSIONS = {
    "application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp",
    "image/gif": "gif", "image/bmp": "bmp", "image/tiff": "tif",
}
_MAX_NAME_ATTEMPTS = 10_000


class StorageError(ValueError):
    """저장 폴더 밖을 가리키는 경로이거나 파일을 쓸 수 없다."""


def extension_for(mime: str, fallback: str = "bin") -> str:
    return _MIME_EXTENSIONS.get(str(mime or "").lower(), fallback)


def safe_filename(name: str, *, max_length: int = 96) -> str:
    """첨부 이름을 탐색기에서 알아볼 수 있는 안전한 파일 이름으로 바꾼다(한글은 그대로 둔다)."""
    text = unicodedata.normalize("NFC", str(name or "")).replace(" · ", ".")
    text = _INVALID_CHARACTERS.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    stem, dot, extension = text.rpartition(".")
    if not dot or not stem or len(extension) > 10:
        stem, extension = text, ""
    stem = stem.strip(" .") or "file"
    if stem.upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    budget = max(8, max_length - (len(extension) + 1 if extension else 0))
    stem = stem[:budget].rstrip(" .") or "file"
    return f"{stem}.{extension}" if extension else stem


class FileStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    # ------------------------------------------------------------------ 경로 검사
    def resolve(self, relative: str) -> Path:
        """상대 경로를 실제 경로로 바꾼다. 저장 폴더 밖을 가리키면 StorageError."""
        text = str(relative or "")
        if not text or "\x00" in text:
            raise StorageError("파일 경로가 비어 있습니다.")
        parts = text.replace("\\", "/").split("/")
        # 빈 조각 = 절대 경로(/a, \\server\share)나 이중 구분자, ":" = 드라이브 문자나 NTFS 대체 스트림
        if any(part in ("", ".", "..") or ":" in part for part in parts):
            raise StorageError(f"저장 폴더 밖을 가리키는 경로는 쓸 수 없습니다: {text!r}")
        root = self.root.resolve()
        path = root.joinpath(*parts).resolve()
        if path == root or not path.is_relative_to(root):
            raise StorageError(f"저장 폴더 밖을 가리키는 경로는 쓸 수 없습니다: {text!r}")
        return path

    def conversation_dir(self, conversation_id: str) -> Path:
        if not isinstance(conversation_id, str) or not _FOLDER_NAME.match(conversation_id):
            raise StorageError(f"대화 ID로 폴더를 만들 수 없습니다: {conversation_id!r}")
        return self.resolve(conversation_id)

    # ------------------------------------------------------------------ 쓰기 / 읽기
    def write(self, conversation_id: str, name: str, data: bytes, *, keep_existing: bool = False) -> str:
        """`{대화ID}/{name}`에 쓰고 저장 폴더 기준 상대 경로를 돌려준다. name은 하위 폴더를 포함할 수 있다.

        같은 이름이 이미 있으면 덮어쓰지 않고 " (2)", " (3)"…을 붙인다.
        keep_existing=True면 이미 있는 파일을 그대로 두고 그 경로를 돌려준다(같은 설정으로 만든 타일 덤프용).
        """
        self.conversation_dir(conversation_id)
        target = f"{conversation_id}/{str(name).replace(chr(92), '/')}"
        self.resolve(target)                  # "/a.png", "../a.png" 같은 이름은 여기서 거부된다
        prefix, _, filename = target.rpartition("/")
        prefix = f"{prefix}/"
        stem, dot, extension = filename.rpartition(".")
        if not dot:
            stem, extension = filename, ""
        for attempt in range(1, _MAX_NAME_ATTEMPTS):
            candidate = filename if attempt == 1 else (f"{stem} ({attempt}).{extension}" if extension else f"{stem} ({attempt})")
            relative = f"{prefix}{candidate}"
            path = self.resolve(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(path, "xb") as handle:
                    handle.write(data)
            except FileExistsError:
                if keep_existing:
                    return relative
                continue
            except OSError as error:
                path.unlink(missing_ok=True)      # 쓰다 만 파일을 남기지 않는다
                raise StorageError(f"첨부 파일을 저장하지 못했습니다({path.name}): {error}") from error
            return relative
        raise StorageError(f"'{filename}'과 같은 이름의 파일이 너무 많습니다.")

    def read(self, relative: str) -> bytes | None:
        """파일이 없거나 읽을 수 없으면 None. 저장 폴더 밖을 가리키는 경로는 StorageError."""
        path = self.resolve(relative)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            logger.warning("첨부 파일이 없습니다: %s", path)
        except OSError as error:
            logger.warning("첨부 파일을 읽지 못했습니다: %s (%s)", path, error)
        return None

    def discard(self, relative: str) -> None:
        """방금 쓴 파일을 되돌린다(DB 저장이 실패했을 때만 쓴다)."""
        try:
            self.resolve(relative).unlink(missing_ok=True)
        except (OSError, StorageError) as error:
            logger.warning("저장에 실패한 첨부 파일을 정리하지 못했습니다: %s (%s)", relative, error)

    # ------------------------------------------------------------------ 삭제
    def remove_conversation(self, conversation_id: str) -> bool:
        """대화 폴더를 통째로 지운다. 실패해도 예외를 던지지 않는다(대화 삭제는 계속 진행돼야 한다)."""
        try:
            folder = self.conversation_dir(conversation_id)
            if not folder.exists():
                return True
            shutil.rmtree(folder)
            return True
        except (OSError, StorageError) as error:
            logger.warning("대화 폴더를 지우지 못했습니다: %s (%s)", conversation_id, error)
            return False
