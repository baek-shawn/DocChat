"""서버 실행: `uv run python run.py`

환경변수: DOCCHAT_HOST(기본 127.0.0.1), DOCCHAT_PORT(기본 8000), DOCCHAT_DB_PATH.
옵션   : --reload (개발용 자동 재시작), --host, --port
"""
from __future__ import annotations

import argparse

import uvicorn

from app import config


def main() -> None:
    parser = argparse.ArgumentParser(description="DocChat 서버")
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--reload", action="store_true", help="코드 변경 시 자동 재시작(개발용)")
    arguments = parser.parse_args()
    if arguments.host not in ("127.0.0.1", "localhost", "::1"):
        print("경고: localhost가 아닌 주소에 바인딩합니다. 이 서버에는 인증이 없으니 신뢰할 수 있는 네트워크에서만 쓰세요.")
    print(f"DocChat: http://{arguments.host}:{arguments.port}")
    print("API key는 요청마다 전달받아 사용하며 디스크에 저장하지 않습니다.")
    uvicorn.run("app.main:app", host=arguments.host, port=arguments.port, reload=arguments.reload, log_level="info")


if __name__ == "__main__":
    main()
