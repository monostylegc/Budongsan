/**
 * 메인 애플리케이션 - 2페이지 구조 (Setup → Simulation)
 */

import { useState } from 'react';
import { useSimulation } from './hooks/useSimulation';
import { SetupPage } from './components/SetupPage';
import { SimulationPage } from './components/SimulationPage';
import { SimulationParams } from './types/simulation';
import './App.css';

type Page = 'setup' | 'simulation';

function App() {
  const [page, setPage] = useState<Page>('setup');

  const {
    status,
    currentMonth,
    totalMonths,
    progress,
    states,
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

  // 시뮬레이션 시작 → 페이지 전환
  const handleStart = (params: SimulationParams) => {
    startSimulation(params);
    setPage('simulation');
  };

  // 설정으로 돌아가기
  const handleBack = () => {
    resetSimulation();
    setPage('setup');
  };

  if (page === 'setup') {
    return <SetupPage onStart={handleStart} />;
  }

  return (
    <SimulationPage
      status={status}
      states={states}
      currentMonth={currentMonth}
      totalMonths={totalMonths}
      progress={progress}
      summary={summary}
      error={error}
      isPlaying={isPlaying}
      playbackMonth={playbackMonth}
      playbackSpeed={playbackSpeed}
      onTogglePlayback={togglePlayback}
      onSetPlaybackMonth={setPlaybackMonth}
      onSetPlaybackSpeed={setPlaybackSpeed}
      onStop={stopSimulation}
      onBack={handleBack}
    />
  );
}

export default App;
