#!/usr/bin/env python
"""웹 시뮬레이션 서버 실행 스크립트"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(__file__).parent
BACKEND_DIR = ROOT_DIR / "web" / "backend"
FRONTEND_DIR = ROOT_DIR / "web" / "frontend"


def check_dependencies():
    """의존성 확인"""
    print("의존성 확인 중...")

    # Python 패키지 확인
    try:
        import fastapi
        import uvicorn
        import websockets
    except ImportError as e:
        print(f"Python 패키지 누락: {e}")
        print("설치 중...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "-r", str(BACKEND_DIR / "requirements.txt")
        ], check=True)

    # Node.js 확인
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        print(f"Node.js: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Node.js가 설치되어 있지 않습니다.")
        print("https://nodejs.org 에서 설치하세요.")
        sys.exit(1)

    # npm 패키지 확인
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("프론트엔드 패키지 설치 중...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True, shell=True)


def run_backend():
    """백엔드 서버 실행"""
    print("\n백엔드 서버 시작 중... (http://localhost:8000)")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=BACKEND_DIR,
        env=env
    )


def run_frontend():
    """프론트엔드 서버 실행"""
    print("\n프론트엔드 서버 시작 중... (http://localhost:3000)")

    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True
    )


def main():
    """메인 실행"""
    print("=" * 60)
    print("한국 부동산 ABM 웹 시뮬레이션")
    print("=" * 60)

    # 의존성 확인
    check_dependencies()

    # 서버 실행
    backend_proc = run_backend()
    time.sleep(3)  # 백엔드 시작 대기

    frontend_proc = run_frontend()
    time.sleep(5)  # 프론트엔드 시작 대기

    # 브라우저 열기
    print("\n" + "=" * 60)
    print("서버가 시작되었습니다!")
    print("프론트엔드: http://localhost:3000")
    print("백엔드 API: http://localhost:8000")
    print("API 문서: http://localhost:8000/docs")
    print("=" * 60)
    print("\n종료하려면 Ctrl+C를 누르세요.\n")

    webbrowser.open("http://localhost:3000")

    try:
        # 프로세스 대기
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n서버 종료 중...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("종료되었습니다.")


if __name__ == "__main__":
    main()
