/**
 * Plotly 차트 패널 - 가격/거래량/지표 시각화
 */

import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { MonthlyState, REGION_NAMES } from '../types/simulation';

interface ChartPanelProps {
  states: MonthlyState[];
  playbackMonth: number;
}

export function ChartPanel({ states, playbackMonth }: ChartPanelProps) {
  // 차트 데이터 준비
  const chartData = useMemo(() => {
    if (states.length === 0) return null;

    const months = states.slice(0, playbackMonth + 1).map((s) => s.month);

    // 지역별 가격 추이
    const gangnamPrices = states.slice(0, playbackMonth + 1).map((s) =>
      s.regions.find((r) => r.region_id === 0)?.price || 0
    );
    const seoulPrices = states.slice(0, playbackMonth + 1).map((s) =>
      s.regions.find((r) => r.region_id === 2)?.price || 0
    );
    const gyeonggiPrices = states.slice(0, playbackMonth + 1).map((s) =>
      s.regions.find((r) => r.region_id === 4)?.price || 0
    );
    const jibangPrices = states.slice(0, playbackMonth + 1).map((s) =>
      s.regions.find((r) => r.region_id === 12)?.price || 0
    );

    // 거래량
    const transactions = states.slice(0, playbackMonth + 1).map((s) => s.total_transactions);

    // 거시지표
    const interestRates = states.slice(0, playbackMonth + 1).map((s) => s.interest_rate * 100);
    const inflations = states.slice(0, playbackMonth + 1).map((s) => s.inflation * 100);
    const homeownerRates = states.slice(0, playbackMonth + 1).map((s) => s.homeowner_rate * 100);

    // 수요/공급 비율
    const dsRatios = states.slice(0, playbackMonth + 1).map((s) => s.demand_supply_ratio);

    return {
      months,
      gangnamPrices,
      seoulPrices,
      gyeonggiPrices,
      jibangPrices,
      transactions,
      interestRates,
      inflations,
      homeownerRates,
      dsRatios,
    };
  }, [states, playbackMonth]);

  if (!chartData || chartData.months.length === 0) {
    return (
      <div className="chart-panel empty">
        <p>시뮬레이션 데이터를 기다리는 중...</p>
      </div>
    );
  }

  // 가격 단위 변환 (억원)
  const toEok = (prices: number[]) => prices.map((p) => p / 10000);

  return (
    <div className="chart-panel">
      {/* 가격 추이 차트 */}
      <div className="chart-container">
        <Plot
          data={[
            {
              x: chartData.months,
              y: toEok(chartData.gangnamPrices),
              type: 'scatter',
              mode: 'lines',
              name: '강남3구',
              line: { color: '#ff4444', width: 2 },
            },
            {
              x: chartData.months,
              y: toEok(chartData.seoulPrices),
              type: 'scatter',
              mode: 'lines',
              name: '기타서울',
              line: { color: '#ff8844', width: 2 },
            },
            {
              x: chartData.months,
              y: toEok(chartData.gyeonggiPrices),
              type: 'scatter',
              mode: 'lines',
              name: '경기남부',
              line: { color: '#44bb44', width: 2 },
            },
            {
              x: chartData.months,
              y: toEok(chartData.jibangPrices),
              type: 'scatter',
              mode: 'lines',
              name: '기타지방',
              line: { color: '#4444ff', width: 2 },
            },
          ]}
          layout={{
            title: '지역별 가격 추이',
            xaxis: { title: '월' },
            yaxis: { title: '가격 (억원)' },
            legend: { x: 0, y: 1.1, orientation: 'h' },
            margin: { t: 50, r: 20, b: 50, l: 60 },
            height: 250,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      {/* 거래량 차트 */}
      <div className="chart-container">
        <Plot
          data={[
            {
              x: chartData.months,
              y: chartData.transactions,
              type: 'bar',
              name: '거래량',
              marker: { color: '#6699ff' },
            },
          ]}
          layout={{
            title: '월별 거래량',
            xaxis: { title: '월' },
            yaxis: { title: '거래 건수' },
            margin: { t: 50, r: 20, b: 50, l: 60 },
            height: 200,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      {/* 거시지표 차트 */}
      <div className="chart-container">
        <Plot
          data={[
            {
              x: chartData.months,
              y: chartData.interestRates,
              type: 'scatter',
              mode: 'lines',
              name: '금리 (%)',
              line: { color: '#ff6644', width: 2 },
              yaxis: 'y',
            },
            {
              x: chartData.months,
              y: chartData.homeownerRates,
              type: 'scatter',
              mode: 'lines',
              name: '자가보유율 (%)',
              line: { color: '#44bb44', width: 2, dash: 'dash' },
              yaxis: 'y2',
            },
          ]}
          layout={{
            title: '거시경제 지표',
            xaxis: { title: '월' },
            yaxis: {
              title: '금리 (%)',
              side: 'left',
              range: [0, 8],
            },
            yaxis2: {
              title: '자가보유율 (%)',
              side: 'right',
              overlaying: 'y',
              range: [40, 70],
            },
            legend: { x: 0, y: 1.15, orientation: 'h' },
            margin: { t: 50, r: 60, b: 50, l: 60 },
            height: 200,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      {/* 수요/공급 비율 */}
      <div className="chart-container">
        <Plot
          data={[
            {
              x: chartData.months,
              y: chartData.dsRatios,
              type: 'scatter',
              mode: 'lines+markers',
              name: '수요/공급 비율',
              line: { color: '#9966ff', width: 2 },
              fill: 'tozeroy',
              fillcolor: 'rgba(153, 102, 255, 0.2)',
            },
            {
              x: chartData.months,
              y: chartData.months.map(() => 1),
              type: 'scatter',
              mode: 'lines',
              name: '균형선',
              line: { color: '#888888', width: 1, dash: 'dot' },
            },
          ]}
          layout={{
            title: '수요/공급 비율',
            xaxis: { title: '월' },
            yaxis: { title: '비율', range: [0, 3] },
            legend: { x: 0, y: 1.1, orientation: 'h' },
            margin: { t: 50, r: 20, b: 50, l: 60 },
            height: 200,
            shapes: [
              {
                type: 'line',
                x0: 0,
                x1: chartData.months[chartData.months.length - 1],
                y0: 1,
                y1: 1,
                line: { color: '#888', width: 1, dash: 'dot' },
              },
            ],
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
