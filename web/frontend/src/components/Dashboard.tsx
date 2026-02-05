/**
 * 대시보드 - 현재 상태 요약
 */

import React from 'react';
import { MonthlyState, SimulationStatus, SimulationSummary } from '../types/simulation';

interface DashboardProps {
  state: MonthlyState | null;
  status: SimulationStatus;
  progress: number;
  currentMonth: number;
  totalMonths: number;
  summary: SimulationSummary | null;
  playbackMonth: number;
  isPlaying: boolean;
  playbackSpeed: number;
  onTogglePlayback: () => void;
  onSetPlaybackMonth: (month: number) => void;
  onSetPlaybackSpeed: (speed: number) => void;
}

// 상태 표시 색상
function getStatusColor(status: SimulationStatus): string {
  switch (status) {
    case 'running': return '#44bb44';
    case 'completed': return '#4488ff';
    case 'error': return '#ff4444';
    case 'stopped': return '#ff8844';
    default: return '#888888';
  }
}

// 상태 표시 텍스트
function getStatusText(status: SimulationStatus): string {
  switch (status) {
    case 'idle': return '대기 중';
    case 'connecting': return '연결 중...';
    case 'initializing': return '초기화 중...';
    case 'running': return '실행 중';
    case 'paused': return '일시정지';
    case 'completed': return '완료';
    case 'error': return '오류';
    case 'stopped': return '중지됨';
  }
}

export function Dashboard({
  state,
  status,
  progress,
  currentMonth,
  totalMonths,
  summary,
  playbackMonth,
  isPlaying,
  playbackSpeed,
  onTogglePlayback,
  onSetPlaybackMonth,
  onSetPlaybackSpeed,
}: DashboardProps) {
  return (
    <div className="dashboard">
      {/* 상태 표시 */}
      <div className="status-bar">
        <div className="status-indicator" style={{ backgroundColor: getStatusColor(status) }}>
          {getStatusText(status)}
        </div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${progress * 100}%` }}
          />
          <span className="progress-text">
            {currentMonth} / {totalMonths} 개월
          </span>
        </div>
      </div>

      {/* 재생 컨트롤 */}
      {currentMonth > 0 && (
        <div className="playback-controls">
          <button
            className={`btn-playback ${isPlaying ? 'playing' : ''}`}
            onClick={onTogglePlayback}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
          <input
            type="range"
            min={0}
            max={currentMonth - 1}
            value={playbackMonth}
            onChange={(e) => onSetPlaybackMonth(Number(e.target.value))}
            className="playback-slider"
          />
          <span className="playback-month">
            {state ? `${state.year}년 ${(playbackMonth % 12) + 1}월` : '-'}
          </span>
          <select
            value={playbackSpeed}
            onChange={(e) => onSetPlaybackSpeed(Number(e.target.value))}
            className="speed-select"
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
          </select>
        </div>
      )}

      {/* 현재 상태 카드 */}
      {state && (
        <div className="stats-cards">
          <div className="stat-card">
            <div className="stat-label">평균 가격</div>
            <div className="stat-value">
              {(state.avg_price / 10000).toFixed(1)}
              <span className="stat-unit">억</span>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-label">월 거래량</div>
            <div className="stat-value">
              {state.total_transactions.toLocaleString()}
              <span className="stat-unit">건</span>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-label">자가보유율</div>
            <div className="stat-value">
              {(state.homeowner_rate * 100).toFixed(1)}
              <span className="stat-unit">%</span>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-label">다주택자 비율</div>
            <div className="stat-value">
              {(state.multi_owner_rate * 100).toFixed(1)}
              <span className="stat-unit">%</span>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-label">기준금리</div>
            <div className="stat-value">
              {(state.interest_rate * 100).toFixed(2)}
              <span className="stat-unit">%</span>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-label">수요/공급</div>
            <div className="stat-value" style={{
              color: state.demand_supply_ratio > 1.2 ? '#ff4444' :
                     state.demand_supply_ratio < 0.8 ? '#4444ff' : '#44bb44'
            }}>
              {state.demand_supply_ratio.toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* 완료 요약 */}
      {summary && status === 'completed' && (
        <div className="summary-panel">
          <h3>시뮬레이션 결과 요약</h3>
          <div className="summary-grid">
            <div className="summary-item">
              <span className="summary-label">기간</span>
              <span className="summary-value">{summary.duration_months}개월</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">강남 가격 변화</span>
              <span className="summary-value" style={{
                color: summary.price_change.gangnam > 0 ? '#ff4444' : '#4444ff'
              }}>
                {summary.price_change.gangnam > 0 ? '+' : ''}
                {(summary.price_change.gangnam * 100).toFixed(1)}%
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">전국 평균 변화</span>
              <span className="summary-value" style={{
                color: summary.price_change.national_avg > 0 ? '#ff4444' : '#4444ff'
              }}>
                {summary.price_change.national_avg > 0 ? '+' : ''}
                {(summary.price_change.national_avg * 100).toFixed(1)}%
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">총 거래</span>
              <span className="summary-value">
                {summary.total_transactions.toLocaleString()}건
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">자가보유율 변화</span>
              <span className="summary-value">
                {(summary.homeowner_rate.initial * 100).toFixed(1)}%
                → {(summary.homeowner_rate.final * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
