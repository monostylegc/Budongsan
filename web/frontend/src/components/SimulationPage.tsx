/**
 * 시뮬레이션 페이지 - 게임맵 + 통계 + 미니차트 + 재생 컨트롤
 */

import { useState } from 'react';
import {
  MonthlyState,
  SimulationStatus,
  SimulationSummary,
} from '../types/simulation';
import { GameMap } from './GameMap';
import { StatsOverview } from './MiniChart';

interface SimulationPageProps {
  // 상태
  status: SimulationStatus;
  states: MonthlyState[];
  currentMonth: number;
  totalMonths: number;
  progress: number;
  summary: SimulationSummary | null;
  error: string | null;

  // 재생
  isPlaying: boolean;
  playbackMonth: number;
  playbackSpeed: number;
  onTogglePlayback: () => void;
  onSetPlaybackMonth: (month: number) => void;
  onSetPlaybackSpeed: (speed: number) => void;

  // 액션
  onStop: () => void;
  onBack: () => void;
}

// 상태 텍스트
function getStatusText(status: SimulationStatus): string {
  switch (status) {
    case 'idle': return '대기';
    case 'connecting': return '연결 중...';
    case 'initializing': return '초기화 중...';
    case 'running': return '실행 중';
    case 'paused': return '일시정지';
    case 'completed': return '완료';
    case 'error': return '오류';
    case 'stopped': return '중지';
  }
}

function getStatusColor(status: SimulationStatus): string {
  switch (status) {
    case 'running': return '#44bb44';
    case 'completed': return '#4488ff';
    case 'error': return '#ff4444';
    case 'stopped': return '#ff8844';
    default: return '#888';
  }
}

export function SimulationPage({
  status, states, currentMonth, totalMonths, progress, summary, error,
  isPlaying, playbackMonth, playbackSpeed,
  onTogglePlayback, onSetPlaybackMonth, onSetPlaybackSpeed,
  onStop, onBack,
}: SimulationPageProps) {
  const [selectedRegion, setSelectedRegion] = useState<number | null>(null);
  const [showSummary, setShowSummary] = useState(true);

  const displayState = states[playbackMonth] || states[states.length - 1] || null;
  const isLoading = status === 'connecting' || status === 'initializing';
  const isRunningOrDone = currentMonth > 0;

  return (
    <div className="sim-page">
      {/* 상단 바 */}
      <header className="sim-header">
        <button className="btn-back" onClick={onBack}>&#8592; 설정</button>

        <div className="sim-header-center">
          <span className="sim-status-dot" style={{ background: getStatusColor(status) }} />
          <span className="sim-status-text">{getStatusText(status)}</span>
          {isRunningOrDone && (
            <div className="sim-progress-bar">
              <div className="sim-progress-fill" style={{ width: `${progress * 100}%` }} />
              <span className="sim-progress-text">{currentMonth}/{totalMonths}</span>
            </div>
          )}
        </div>

        <div className="sim-header-right">
          {(status === 'running' || status === 'connecting' || status === 'initializing') && (
            <button className="btn-stop-sim" onClick={onStop}>중지</button>
          )}
        </div>
      </header>

      {/* 에러 메시지 */}
      {error && (
        <div className="sim-error">
          <span>&#9888; {error}</span>
        </div>
      )}

      {/* 로딩 오버레이 */}
      {isLoading && (
        <div className="sim-loading-overlay">
          <div className="loading-spinner" />
          <p>{status === 'connecting' ? '서버 연결 중...' : '시뮬레이션 초기화 중...'}</p>
          <p className="loading-hint">에이전트 초기화에 약 10-30초 소요</p>
        </div>
      )}

      {/* 메인 콘텐츠 */}
      <div className="sim-content">
        {/* 좌: 게임맵 */}
        <div className="sim-map-panel">
          <GameMap
            state={displayState}
            onSelectRegion={setSelectedRegion}
            selectedRegion={selectedRegion}
          />
        </div>

        {/* 우: 통계 + 차트 */}
        <div className="sim-stats-panel">
          <StatsOverview
            states={states}
            currentIdx={playbackMonth}
            selectedRegion={selectedRegion}
          />
        </div>
      </div>

      {/* 하단 재생 컨트롤 */}
      {isRunningOrDone && (
        <footer className="sim-playback">
          <button
            className={`btn-play ${isPlaying ? 'playing' : ''}`}
            onClick={onTogglePlayback}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>

          <input
            type="range"
            className="playback-slider"
            min={0}
            max={Math.max(0, currentMonth - 1)}
            value={playbackMonth}
            onChange={(e) => onSetPlaybackMonth(Number(e.target.value))}
          />

          <span className="playback-time">
            {displayState
              ? `${displayState.year}년 ${(playbackMonth % 12) + 1}월`
              : '-'}
          </span>

          <select
            className="speed-select"
            value={playbackSpeed}
            onChange={(e) => onSetPlaybackSpeed(Number(e.target.value))}
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
            <option value={8}>8x</option>
          </select>
        </footer>
      )}

      {/* 완료 요약 */}
      {summary && status === 'completed' && showSummary && (
        <div className="sim-summary-overlay">
          <div className="sim-summary-card">
            <h3>시뮬레이션 완료</h3>
            <div className="summary-grid">
              <div><span>기간</span><span>{summary.duration_months}개월</span></div>
              <div>
                <span>강남 변화</span>
                <span style={{ color: summary.price_change.gangnam > 0 ? '#ff4444' : '#4488ff' }}>
                  {summary.price_change.gangnam > 0 ? '+' : ''}{(summary.price_change.gangnam * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span>전국 평균</span>
                <span style={{ color: summary.price_change.national_avg > 0 ? '#ff4444' : '#4488ff' }}>
                  {summary.price_change.national_avg > 0 ? '+' : ''}{(summary.price_change.national_avg * 100).toFixed(1)}%
                </span>
              </div>
              <div><span>총 거래</span><span>{summary.total_transactions.toLocaleString()}건</span></div>
              <div>
                <span>자가율</span>
                <span>{(summary.homeowner_rate.initial * 100).toFixed(1)}% → {(summary.homeowner_rate.final * 100).toFixed(1)}%</span>
              </div>
            </div>
            <button className="btn-close-summary" onClick={() => setShowSummary(false)}>
              확인
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
