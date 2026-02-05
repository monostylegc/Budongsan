/**
 * 시뮬레이션 WebSocket 연결 및 상태 관리 훅
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  SimulationParams,
  MonthlyState,
  SimulationStatus,
  WSMessage,
  SimulationSummary,
} from '../types/simulation';

interface UseSimulationReturn {
  // 상태
  status: SimulationStatus;
  currentMonth: number;
  totalMonths: number;
  progress: number;
  states: MonthlyState[];
  currentState: MonthlyState | null;
  summary: SimulationSummary | null;
  error: string | null;
  sessionId: string | null;

  // 액션
  startSimulation: (params: SimulationParams) => Promise<void>;
  stopSimulation: () => void;
  resetSimulation: () => void;

  // 재생 제어
  isPlaying: boolean;
  playbackMonth: number;
  setPlaybackMonth: (month: number) => void;
  togglePlayback: () => void;
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
}

const API_BASE = '';  // 프록시 사용

export function useSimulation(): UseSimulationReturn {
  // 상태
  const [status, setStatus] = useState<SimulationStatus>('idle');
  const [states, setStates] = useState<MonthlyState[]>([]);
  const [summary, setSummary] = useState<SimulationSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [totalMonths, setTotalMonths] = useState(120);

  // 재생 제어
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackMonth, setPlaybackMonth] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const playbackIntervalRef = useRef<number | null>(null);

  // 현재 월과 상태
  const currentMonth = states.length;
  const progress = totalMonths > 0 ? currentMonth / totalMonths : 0;
  const currentState = states.length > 0 ? states[playbackMonth] || states[states.length - 1] : null;

  // 시뮬레이션 시작
  const startSimulation = useCallback(async (params: SimulationParams) => {
    setError(null);
    setStates([]);
    setSummary(null);
    setPlaybackMonth(0);
    setIsPlaying(false);
    setTotalMonths(params.num_steps);
    setStatus('connecting');

    try {
      // 세션 생성
      const response = await fetch(`${API_BASE}/api/simulation/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        throw new Error('시뮬레이션 시작 실패');
      }

      const data = await response.json();
      const newSessionId = data.session_id;
      setSessionId(newSessionId);

      // WebSocket 연결
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = window.location.host;
      const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/simulation/${newSessionId}`);

      ws.onopen = () => {
        setStatus('initializing');
      };

      ws.onmessage = (event) => {
        const message: WSMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'status':
            if (message.message?.includes('시작')) {
              setStatus('running');
            }
            break;

          case 'state':
            if (message.data) {
              setStates((prev) => [...prev, message.data!]);
              // 실시간 모드: 최신 상태 표시
              if (!isPlaying) {
                setPlaybackMonth((prev) => prev + 1);
              }
            }
            break;

          case 'completed':
            setStatus('completed');
            if (message.summary) {
              setSummary(message.summary);
            }
            break;

          case 'stopped':
            setStatus('stopped');
            break;

          case 'error':
            setStatus('error');
            setError(message.message || '알 수 없는 오류');
            break;
        }
      };

      ws.onerror = () => {
        setStatus('error');
        setError('WebSocket 연결 오류');
      };

      ws.onclose = () => {
        if (status === 'running') {
          setStatus('error');
          setError('연결이 끊어졌습니다.');
        }
      };

      wsRef.current = ws;

    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : '시뮬레이션 시작 실패');
    }
  }, [status, isPlaying]);

  // 시뮬레이션 중지
  const stopSimulation = useCallback(() => {
    if (sessionId) {
      fetch(`${API_BASE}/api/simulation/${sessionId}/stop`, { method: 'POST' });
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus('stopped');
    setIsPlaying(false);
  }, [sessionId]);

  // 시뮬레이션 리셋
  const resetSimulation = useCallback(() => {
    stopSimulation();
    setStates([]);
    setSummary(null);
    setError(null);
    setSessionId(null);
    setPlaybackMonth(0);
    setStatus('idle');
  }, [stopSimulation]);

  // 재생 토글
  const togglePlayback = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // 재생 효과
  useEffect(() => {
    if (isPlaying && states.length > 0) {
      const interval = 1000 / playbackSpeed;
      playbackIntervalRef.current = window.setInterval(() => {
        setPlaybackMonth((prev) => {
          if (prev >= states.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, interval);
    } else {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
        playbackIntervalRef.current = null;
      }
    }

    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, states.length]);

  // 정리
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, []);

  return {
    status,
    currentMonth,
    totalMonths,
    progress,
    states,
    currentState,
    summary,
    error,
    sessionId,
    startSimulation,
    stopSimulation,
    resetSimulation,
    isPlaying,
    playbackMonth,
    setPlaybackMonth,
    togglePlayback,
    playbackSpeed,
    setPlaybackSpeed,
  };
}
