# Python 백엔드 Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY web/backend/requirements.txt /app/web/backend/
RUN pip install --no-cache-dir -r /app/web/backend/requirements.txt

# 소스 코드 복사
COPY src/ /app/src/
COPY web/backend/ /app/web/backend/

# 작업 디렉토리 설정
WORKDIR /app/web/backend

# 포트 노출
EXPOSE 8000

# 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
