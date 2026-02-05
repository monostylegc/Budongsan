"""FastAPI 백엔드 - 한국 부동산 ABM 웹 인터랙티브 시각화"""

import asyncio
import uuid
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    SimulationParams,
    SimulationStartResponse,
    SimulationStatusResponse,
    DefaultParamsResponse,
    ScenarioPreset,
    ScenarioType,
)
from sim_runner import SimulationRunner, get_scenario_presets


# 세션 저장소
sessions: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 수명주기 관리"""
    # 시작 시
    print("[Server] 부동산 ABM 시뮬레이션 서버 시작")
    yield
    # 종료 시
    print("서버 종료 중...")
    # 모든 세션 정리
    for session_id, session in sessions.items():
        runner = session.get('runner')
        if runner and runner.is_running:
            runner.stop()
    sessions.clear()


app = FastAPI(
    title="한국 부동산 ABM 시뮬레이션 API",
    description="행동경제학 기반 한국 부동산 시장 Agent-Based Model",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "한국 부동산 ABM 시뮬레이션 API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/params/defaults", response_model=DefaultParamsResponse)
async def get_default_params():
    """기본 파라미터 및 시나리오 프리셋 반환"""
    presets = get_scenario_presets()

    scenarios = {}
    for key, preset in presets.items():
        base_params = SimulationParams().model_dump()
        # 시나리오 파라미터 병합
        for category, values in preset.get('params', {}).items():
            if category in base_params:
                if isinstance(base_params[category], dict):
                    base_params[category].update(values)
                else:
                    base_params[category] = values

        scenarios[key] = ScenarioPreset(
            name=preset['name'],
            description=preset['description'],
            params=SimulationParams(**base_params),
        )

    return DefaultParamsResponse(
        params=SimulationParams(),
        scenarios=scenarios,
    )


@app.get("/api/params/scenarios")
async def get_scenarios():
    """시나리오 프리셋 목록 반환"""
    presets = get_scenario_presets()
    return {
        "scenarios": [
            {
                "id": key,
                "name": preset['name'],
                "description": preset['description'],
            }
            for key, preset in presets.items()
        ]
    }


@app.post("/api/simulation/start", response_model=SimulationStartResponse)
async def start_simulation(params: SimulationParams):
    """시뮬레이션 시작

    새 세션을 생성하고 session_id를 반환합니다.
    WebSocket으로 연결하여 실시간 결과를 받으세요.
    """
    session_id = str(uuid.uuid4())[:8]

    # 파라미터 딕셔너리로 변환
    params_dict = params.model_dump()

    # 시뮬레이션 러너 생성 (아직 초기화하지 않음 - WebSocket 연결 시 초기화)
    sessions[session_id] = {
        'params': params_dict,
        'runner': None,
        'status': 'pending',
        'current_step': 0,
        'total_steps': params.num_steps,
    }

    return SimulationStartResponse(
        session_id=session_id,
        status="pending",
        message=f"세션 생성됨. WebSocket /ws/simulation/{session_id} 로 연결하세요.",
        params=params,
    )


@app.get("/api/simulation/{session_id}/status", response_model=SimulationStatusResponse)
async def get_simulation_status(session_id: str):
    """시뮬레이션 상태 조회"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]
    runner = session.get('runner')

    status = session.get('status', 'pending')
    current_step = 0
    if runner and runner.sim:
        current_step = runner.sim.current_step

    return SimulationStatusResponse(
        session_id=session_id,
        status=status,
        current_step=current_step,
        total_steps=session.get('total_steps', 120),
        progress=current_step / session.get('total_steps', 120) if session.get('total_steps') else 0,
    )


@app.post("/api/simulation/{session_id}/stop")
async def stop_simulation(session_id: str):
    """시뮬레이션 중지"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]
    runner = session.get('runner')

    if runner and runner.is_running:
        runner.stop()
        session['status'] = 'stopped'
        return {"message": "시뮬레이션 중지 요청됨", "session_id": session_id}

    return {"message": "시뮬레이션이 실행 중이 아닙니다.", "session_id": session_id}


@app.delete("/api/simulation/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions.pop(session_id)
    runner = session.get('runner')
    if runner and runner.is_running:
        runner.stop()

    return {"message": "세션 삭제됨", "session_id": session_id}


@app.websocket("/ws/simulation/{session_id}")
async def simulation_stream(websocket: WebSocket, session_id: str):
    """시뮬레이션 실시간 스트리밍

    WebSocket으로 매월 결과를 전송합니다.
    메시지 형식:
    - {"type": "state", "data": {...}}: 월별 상태
    - {"type": "completed", "summary": {...}}: 완료
    - {"type": "error", "message": "..."}: 오류
    - {"type": "stopped", "month": N}: 중지됨
    """
    if session_id not in sessions:
        await websocket.close(code=4004, reason="세션을 찾을 수 없습니다.")
        return

    await websocket.accept()

    session = sessions[session_id]
    params = session['params']

    try:
        # 시뮬레이션 러너 생성 및 초기화
        await websocket.send_json({
            'type': 'status',
            'message': '시뮬레이션 초기화 중...',
        })

        # GPU 가속 시도, 실패 시 CPU로 폴백
        try:
            runner = SimulationRunner(params, arch="vulkan")
        except Exception:
            try:
                runner = SimulationRunner(params, arch="cuda")
            except Exception:
                runner = SimulationRunner(params, arch="cpu")

        session['runner'] = runner
        session['status'] = 'initializing'

        # 초기화
        runner.initialize()
        session['status'] = 'running'

        await websocket.send_json({
            'type': 'status',
            'message': '시뮬레이션 시작',
        })

        # 스트리밍 실행
        await runner.run_streaming(websocket)

        session['status'] = 'completed'

    except WebSocketDisconnect:
        print(f"WebSocket 연결 끊김: {session_id}")
        if session.get('runner'):
            session['runner'].stop()
        session['status'] = 'disconnected'

    except Exception as e:
        print(f"시뮬레이션 오류: {e}")
        session['status'] = 'error'
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e),
            })
        except Exception:
            pass


@app.get("/api/regions")
async def get_regions():
    """지역 정보 반환"""
    from src.realestate.config import REGIONS

    return {
        "regions": [
            {
                "id": region_id,
                "name": info['name'],
                "tier": info['tier'],
                "base_price": info['base_price'],
            }
            for region_id, info in REGIONS.items()
        ]
    }


# 개발용 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
