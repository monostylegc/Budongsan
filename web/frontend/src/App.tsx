/**
 * 메인 애플리케이션 컴포넌트
 */

import React, { useState } from 'react';
import { useSimulation } from './hooks/useSimulation';
import { ParameterPanel } from './components/ParameterPanel';
import { MapView } from './components/MapView';
import { ChartPanel } from './components/ChartPanel';
import { Dashboard } from './components/Dashboard';
import { SimulationControl } from './components/SimulationControl';
import './App.css';

function App() {
  // 시뮬레이션 상태
  const {
    status,
    currentMonth,
    totalMonths,
    progress,
    states,
    currentState,
    summary,
    error,
    startSimulation,
    stopSimulation,
    resetSimulation,
    isPlaying,
    playbackMonth,
    setPlaybackMonth,
    togglePlayback,
    playbackSpeed,
    setPlaybackSpeed,
  } = useSimulation();

  // 표시 옵션
  const [showAgentDist, setShowAgentDist] = useState(false);
  const [showTransactions, setShowTransactions] = useState(true);
  const [activeView, setActiveView] = useState<'map' | 'chart' | 'both'>('both');

  // 현재 표시 상태 (재생 위치 기준)
  const displayState = states[playbackMonth] || currentState;

  const isRunning = status === 'running' || status === 'connecting' || status === 'initializing';
  const isCompleted = status === 'completed' || status === 'stopped';

  return (
    <div className="app">
      {/* 헤더 */}
      <header className="app-header">
        <h1>한국 부동산 ABM 시뮬레이션</h1>
        <div className="header-subtitle">
          행동경제학 기반 Agent-Based Model
        </div>
      </header>

      <div className="app-content">
        {/* 좌측 사이드바: 파라미터 패널 */}
        <aside className="sidebar">
          <ParameterPanel
            onStart={startSimulation}
            onStop={stopSimulation}
            onReset={resetSimulation}
            isRunning={isRunning}
            isCompleted={isCompleted}
          />
        </aside>

        {/* 메인 영역 */}
        <main className="main-area">
          {/* 대시보드 */}
          <Dashboard
            state={displayState}
            status={status}
            progress={progress}
            currentMonth={currentMonth}
            totalMonths={totalMonths}
            summary={summary}
            playbackMonth={playbackMonth}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
            onTogglePlayback={togglePlayback}
            onSetPlaybackMonth={setPlaybackMonth}
            onSetPlaybackSpeed={setPlaybackSpeed}
          />

          {/* 제어 오버레이 */}
          <SimulationControl status={status} error={error} />

          {/* 뷰 전환 탭 */}
          <div className="view-tabs">
            <button
              className={activeView === 'both' ? 'active' : ''}
              onClick={() => setActiveView('both')}
            >
              지도 + 차트
            </button>
            <button
              className={activeView === 'map' ? 'active' : ''}
              onClick={() => setActiveView('map')}
            >
              지도
            </button>
            <button
              className={activeView === 'chart' ? 'active' : ''}
              onClick={() => setActiveView('chart')}
            >
              차트
            </button>

            <div className="view-options">
              <label>
                <input
                  type="checkbox"
                  checked={showAgentDist}
                  onChange={(e) => setShowAgentDist(e.target.checked)}
                />
                에이전트 분포
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={showTransactions}
                  onChange={(e) => setShowTransactions(e.target.checked)}
                />
                거래 표시
              </label>
            </div>
          </div>

          {/* 시각화 영역 */}
          <div className={`visualization-area view-${activeView}`}>
            {(activeView === 'map' || activeView === 'both') && (
              <div className="map-section">
                <MapView
                  state={displayState}
                  showAgentDist={showAgentDist}
                  showTransactions={showTransactions}
                />
              </div>
            )}

            {(activeView === 'chart' || activeView === 'both') && (
              <div className="chart-section">
                <ChartPanel
                  states={states}
                  playbackMonth={playbackMonth}
                />
              </div>
            )}
          </div>
        </main>
      </div>

      {/* 푸터 */}
      <footer className="app-footer">
        <span>한국 부동산 ABM v1.0</span>
        <span>|</span>
        <span>Taichi GPU 가속</span>
        <span>|</span>
        <span>행동경제학 기반 의사결정 모델</span>
      </footer>
    </div>
  );
}

export default App;
