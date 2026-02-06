/**
 * 미니차트 컴포넌트 - Plotly 대체 순수 SVG 차트
 * LineChart, BarChart, StatCard
 */

import { useMemo } from 'react';
import { MonthlyState } from '../types/simulation';

// ─── 공통 유틸 ───

function formatEok(value: number): string {
  return (value / 10000).toFixed(1);
}

function formatPct(value: number, decimals = 1): string {
  return (value * 100).toFixed(decimals);
}

// ─── StatCard ───

interface StatCardProps {
  label: string;
  value: string;
  unit?: string;
  change?: number; // 전월 대비 변화 (양수=상승)
  color?: string;
}

export function StatCard({ label, value, unit, change, color }: StatCardProps) {
  return (
    <div className="mini-stat-card">
      <div className="mini-stat-label">{label}</div>
      <div className="mini-stat-value" style={{ color: color || '#eee' }}>
        {value}
        {unit && <span className="mini-stat-unit">{unit}</span>}
      </div>
      {change !== undefined && (
        <div className={`mini-stat-change ${change > 0 ? 'up' : change < 0 ? 'down' : 'flat'}`}>
          {change > 0 ? '▲' : change < 0 ? '▼' : '─'} {Math.abs(change).toFixed(1)}%
        </div>
      )}
    </div>
  );
}

// ─── LineChart (SVG) ───

interface LineChartProps {
  title: string;
  states: MonthlyState[];
  endIdx: number;
  lines: {
    label: string;
    color: string;
    getValue: (s: MonthlyState) => number;
  }[];
  formatY?: (v: number) => string;
  height?: number;
}

export function LineChart({ title, states, endIdx, lines, formatY, height = 120 }: LineChartProps) {
  const chartData = useMemo(() => {
    if (states.length === 0 || endIdx < 0) return null;

    const slice = states.slice(0, endIdx + 1);
    const n = slice.length;
    if (n === 0) return null;

    // 각 라인 데이터 + Y축 범위
    let yMin = Infinity, yMax = -Infinity;
    const series = lines.map(line => {
      const values = slice.map(s => line.getValue(s));
      for (const v of values) {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
      return { ...line, values };
    });

    if (yMax === yMin) { yMax = yMin + 1; }
    const yPad = (yMax - yMin) * 0.1;
    yMin -= yPad;
    yMax += yPad;

    return { series, n, yMin, yMax };
  }, [states, endIdx, lines]);

  if (!chartData) {
    return (
      <div className="mini-chart-container">
        <div className="mini-chart-title">{title}</div>
        <div className="mini-chart-empty">데이터 대기 중...</div>
      </div>
    );
  }

  const { series, n, yMin, yMax } = chartData;
  const svgW = 300;
  const svgH = height;
  const padL = 6, padR = 6, padT = 4, padB = 4;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;

  const toX = (i: number) => padL + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const toY = (v: number) => padT + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  return (
    <div className="mini-chart-container">
      <div className="mini-chart-header">
        <span className="mini-chart-title">{title}</span>
        <div className="mini-chart-legend">
          {series.map(s => (
            <span key={s.label} className="legend-item">
              <span className="legend-dot" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="mini-chart-svg">
        {/* Y축 기준선 */}
        {[0.25, 0.5, 0.75].map(ratio => {
          const y = padT + ratio * plotH;
          return (
            <line key={ratio} x1={padL} y1={y} x2={svgW - padR} y2={y}
              stroke="rgba(100,100,100,0.2)" strokeWidth={0.5} />
          );
        })}

        {/* 라인 */}
        {series.map(s => {
          const points = s.values.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
          return (
            <g key={s.label}>
              <polyline points={points} fill="none" stroke={s.color} strokeWidth={1.5}
                strokeLinecap="round" strokeLinejoin="round" />
            </g>
          );
        })}

        {/* Y축 라벨 */}
        <text x={padL + 2} y={padT + 10} fill="#888" fontSize={8}>
          {formatY ? formatY(yMax) : yMax.toFixed(1)}
        </text>
        <text x={padL + 2} y={svgH - padB - 2} fill="#888" fontSize={8}>
          {formatY ? formatY(yMin) : yMin.toFixed(1)}
        </text>
      </svg>
    </div>
  );
}

// ─── BarChart (SVG) ───

interface BarChartProps {
  title: string;
  states: MonthlyState[];
  endIdx: number;
  getValue: (s: MonthlyState) => number;
  color?: string;
  height?: number;
}

export function BarChart({ title, states, endIdx, getValue, color = '#6699ff', height = 80 }: BarChartProps) {
  const chartData = useMemo(() => {
    if (states.length === 0 || endIdx < 0) return null;
    const slice = states.slice(0, endIdx + 1);
    const values = slice.map(s => getValue(s));
    const maxV = Math.max(...values, 1);
    return { values, maxV };
  }, [states, endIdx, getValue]);

  if (!chartData) return null;

  const { values, maxV } = chartData;
  const svgW = 300;
  const svgH = height;
  const padL = 4, padR = 4, padT = 4, padB = 4;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;
  const barW = Math.max(1, plotW / values.length - 1);

  return (
    <div className="mini-chart-container">
      <div className="mini-chart-title">{title}</div>
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="mini-chart-svg">
        {values.map((v, i) => {
          const barH = (v / maxV) * plotH;
          const x = padL + i * (plotW / values.length);
          const y = padT + plotH - barH;
          return (
            <rect key={i} x={x} y={y} width={barW} height={barH}
              fill={color} opacity={0.7} rx={0.5} />
          );
        })}
      </svg>
    </div>
  );
}

// ─── 통계 패널 (통합) ───

interface StatsOverviewProps {
  states: MonthlyState[];
  currentIdx: number;
  selectedRegion: number | null;
}

export function StatsOverview({ states, currentIdx, selectedRegion }: StatsOverviewProps) {
  const current = states[currentIdx] || null;
  const prev = currentIdx > 0 ? states[currentIdx - 1] : null;

  if (!current) {
    return <div className="stats-overview empty">시뮬레이션 데이터 대기 중...</div>;
  }

  // 선택된 지역 데이터
  const regionData = selectedRegion !== null
    ? current.regions.find(r => r.region_id === selectedRegion)
    : null;

  // 전월 대비 변화 계산
  const avgPriceChange = prev ? ((current.avg_price - prev.avg_price) / prev.avg_price) * 100 : 0;
  const transChange = prev ? ((current.total_transactions - prev.total_transactions) / (prev.total_transactions || 1)) * 100 : 0;

  return (
    <div className="stats-overview">
      {/* 핵심 통계 카드 */}
      <div className="stats-grid">
        <StatCard label="평균가" value={formatEok(current.avg_price)} unit="억"
          change={avgPriceChange} />
        <StatCard label="거래" value={current.total_transactions.toLocaleString()} unit="건"
          change={transChange} />
        <StatCard label="자가율" value={formatPct(current.homeowner_rate)} unit="%"
          color={current.homeowner_rate > 0.55 ? '#44dd66' : '#ffcc33'} />
        <StatCard label="실업률" value={(current.unemployment_rate * 100).toFixed(1)} unit="%"
          color={current.unemployment_rate > 0.04 ? '#ff6644' : '#44dd66'} />
        <StatCard label="금리" value={(current.interest_rate * 100).toFixed(2)} unit="%" />
        <StatCard label="수요/공급" value={current.demand_supply_ratio.toFixed(2)}
          color={current.demand_supply_ratio > 1.2 ? '#ff6644' : current.demand_supply_ratio < 0.8 ? '#4488ff' : '#44dd66'} />
      </div>

      {/* 선택 지역 상세 */}
      {regionData && (
        <div className="region-detail">
          <div className="region-detail-title">{regionData.name}</div>
          <div className="region-detail-grid">
            <div><span>가격</span><span>{formatEok(regionData.price)}억</span></div>
            <div><span>변화</span><span style={{ color: regionData.price_change > 0 ? '#ff6644' : '#44aaff' }}>
              {(regionData.price_change * 100).toFixed(1)}%</span></div>
            <div><span>거래</span><span>{regionData.transactions}건</span></div>
            <div><span>수요</span><span>{regionData.demand}</span></div>
            <div><span>공급</span><span>{regionData.supply}</span></div>
          </div>
        </div>
      )}

      {/* 미니 차트들 */}
      <LineChart
        title="지역별 가격 추이"
        states={states}
        endIdx={currentIdx}
        lines={[
          { label: '강남', color: '#ff4444', getValue: s => (s.regions.find(r => r.region_id === 0)?.price || 0) / 10000 },
          { label: '서울', color: '#ff8844', getValue: s => (s.regions.find(r => r.region_id === 2)?.price || 0) / 10000 },
          { label: '경기', color: '#44bb44', getValue: s => (s.regions.find(r => r.region_id === 4)?.price || 0) / 10000 },
          { label: '지방', color: '#4488ff', getValue: s => (s.regions.find(r => r.region_id === 12)?.price || 0) / 10000 },
        ]}
        formatY={v => v.toFixed(1) + '억'}
      />

      <BarChart
        title="월별 거래량"
        states={states}
        endIdx={currentIdx}
        getValue={s => s.total_transactions}
      />

      <LineChart
        title="거시지표"
        states={states}
        endIdx={currentIdx}
        lines={[
          { label: '금리', color: '#ff6644', getValue: s => s.interest_rate * 100 },
          { label: '자가율', color: '#44bb44', getValue: s => s.homeowner_rate * 100 },
        ]}
        formatY={v => v.toFixed(1) + '%'}
        height={100}
      />
    </div>
  );
}
