/**
 * 게임맵 컴포넌트 - Canvas 2D 기반 한반도 타일맵
 * 13개 지역을 Game of Life 스타일로 시각화
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { MonthlyState, RegionStats, TILE_LAYOUT, ADJACENCY } from '../types/simulation';

interface GameMapProps {
  state: MonthlyState | null;
  onSelectRegion?: (regionId: number | null) => void;
  selectedRegion?: number | null;
}

// 타일 위치 계산용 상수
const GRID_COLS = 5;
const GRID_ROWS = 6;
const TILE_PAD = 6;

// 가격 히트맵 색상 (파랑 → 빨강)
function priceToColor(price: number, minP: number, maxP: number): string {
  if (maxP <= minP) return 'hsl(200, 60%, 40%)';
  const ratio = Math.max(0, Math.min(1, (price - minP) / (maxP - minP)));
  // 파랑(200) → 빨강(0)
  const hue = 200 - ratio * 200;
  return `hsl(${hue}, 70%, ${35 + ratio * 15}%)`;
}

// 가격 변화 테두리 색상
function changeToGlow(change: number): string {
  if (change > 0.02) return 'rgba(255, 60, 60, 0.8)';
  if (change > 0.005) return 'rgba(255, 150, 50, 0.6)';
  if (change > -0.005) return 'rgba(100, 200, 100, 0.3)';
  if (change > -0.02) return 'rgba(50, 150, 255, 0.6)';
  return 'rgba(50, 50, 255, 0.8)';
}

// 에이전트 파티클 생성 (지역별 샘플링)
interface Particle {
  x: number;
  y: number;
  color: string;
}

function sampleParticles(region: RegionStats, tileX: number, tileY: number, tileW: number, tileH: number): Particle[] {
  const particles: Particle[] = [];
  const total = region.homeless_count + region.one_house_count + region.multi_house_count;
  if (total === 0) return particles;

  const maxParticles = 60;
  const scale = Math.min(maxParticles / total, 1);

  const types: { count: number; color: string }[] = [
    { count: region.one_house_count, color: '#44dd66' },    // 초록 = 유주택
    { count: region.homeless_count, color: '#ffcc33' },      // 노랑 = 무주택
    { count: region.multi_house_count, color: '#ff6644' },   // 빨강 = 다주택
  ];

  // 시드 기반 의사 난수 (지역 ID를 시드로)
  let seed = region.region_id * 1000 + 7;
  const rng = () => { seed = (seed * 16807 + 0) % 2147483647; return seed / 2147483647; };

  const margin = 8;
  for (const type of types) {
    const n = Math.max(1, Math.round(type.count * scale));
    for (let i = 0; i < n; i++) {
      particles.push({
        x: tileX + margin + rng() * (tileW - margin * 2),
        y: tileY + 22 + rng() * (tileH - 34),
        color: type.color,
      });
    }
  }
  return particles;
}

export function GameMap({ state, onSelectRegion, selectedRegion }: GameMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredRegion, setHoveredRegion] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; region: RegionStats } | null>(null);
  const [canvasSize, setCanvasSize] = useState({ w: 600, h: 500 });

  // 캔버스 크기 조정
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setCanvasSize({ w: Math.floor(width), h: Math.floor(height) });
        }
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // 타일 사각형 계산
  const getTileRects = useCallback(() => {
    const { w, h } = canvasSize;
    const tileW = (w - TILE_PAD * (GRID_COLS + 1)) / GRID_COLS;
    const tileH = (h - TILE_PAD * (GRID_ROWS + 1)) / GRID_ROWS;

    return TILE_LAYOUT.map(tile => ({
      ...tile,
      rx: TILE_PAD + tile.col * (tileW + TILE_PAD),
      ry: TILE_PAD + tile.row * (tileH + TILE_PAD),
      rw: tileW,
      rh: tileH,
    }));
  }, [canvasSize]);

  // 마우스 이벤트 핸들러
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !state) return;

    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    const tiles = getTileRects();
    let found: number | null = null;

    for (const t of tiles) {
      if (mx >= t.rx && mx <= t.rx + t.rw && my >= t.ry && my <= t.ry + t.rh) {
        found = t.id;
        const regionData = state.regions.find(r => r.region_id === t.id);
        if (regionData) {
          setTooltip({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
            region: regionData,
          });
        }
        break;
      }
    }
    setHoveredRegion(found);
    if (!found) setTooltip(null);
  }, [state, getTileRects]);

  const handleMouseLeave = useCallback(() => {
    setHoveredRegion(null);
    setTooltip(null);
  }, []);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !onSelectRegion) return;

    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    const tiles = getTileRects();
    for (const t of tiles) {
      if (mx >= t.rx && mx <= t.rx + t.rw && my >= t.ry && my <= t.ry + t.rh) {
        onSelectRegion(selectedRegion === t.id ? null : t.id);
        return;
      }
    }
    onSelectRegion(null);
  }, [getTileRects, onSelectRegion, selectedRegion]);

  // 메인 렌더링
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasSize.w * dpr;
    canvas.height = canvasSize.h * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    // 배경
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, canvasSize.w, canvasSize.h);

    // 격자 패턴
    ctx.strokeStyle = 'rgba(48, 54, 61, 0.4)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < canvasSize.w; x += 20) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvasSize.h); ctx.stroke();
    }
    for (let y = 0; y < canvasSize.h; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasSize.w, y); ctx.stroke();
    }

    const tiles = getTileRects();
    const tileMap = new Map<number, typeof tiles[0]>();
    tiles.forEach(t => tileMap.set(t.id, t));

    // 가격 범위
    let minP = Infinity, maxP = -Infinity;
    if (state) {
      for (const r of state.regions) {
        if (r.price < minP) minP = r.price;
        if (r.price > maxP) maxP = r.price;
      }
    }

    // 인접 연결선 (풍선효과)
    ctx.lineWidth = 1.5;
    for (const [a, b] of ADJACENCY) {
      const ta = tileMap.get(a);
      const tb = tileMap.get(b);
      if (!ta || !tb) continue;

      const ax = ta.rx + ta.rw / 2;
      const ay = ta.ry + ta.rh / 2;
      const bx = tb.rx + tb.rw / 2;
      const by = tb.ry + tb.rh / 2;

      // 가격 전파가 있으면 빛나는 선
      if (state) {
        const ra = state.regions.find(r => r.region_id === a);
        const rb = state.regions.find(r => r.region_id === b);
        if (ra && rb) {
          const spread = Math.abs(ra.price_change - rb.price_change);
          const alpha = Math.min(0.7, spread * 10 + 0.08);
          ctx.strokeStyle = `rgba(100, 180, 255, ${alpha})`;
        } else {
          ctx.strokeStyle = 'rgba(60, 70, 80, 0.3)';
        }
      } else {
        ctx.strokeStyle = 'rgba(60, 70, 80, 0.3)';
      }

      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }

    // 타일 그리기
    for (const tile of tiles) {
      const regionData = state?.regions.find(r => r.region_id === tile.id);
      const isHovered = hoveredRegion === tile.id;
      const isSelected = selectedRegion === tile.id;

      // 타일 배경
      const bgColor = regionData
        ? priceToColor(regionData.price, minP, maxP)
        : 'hsl(210, 20%, 25%)';

      // 둥근 사각형
      const r = 6;
      ctx.beginPath();
      ctx.roundRect(tile.rx, tile.ry, tile.rw, tile.rh, r);
      ctx.fillStyle = bgColor;
      ctx.fill();

      // 테두리 (가격 변화 기반 글로우)
      if (regionData) {
        ctx.strokeStyle = isSelected ? '#fff' : isHovered ? '#88ccff' : changeToGlow(regionData.price_change);
        ctx.lineWidth = isSelected ? 2.5 : isHovered ? 2 : 1.5;
      } else {
        ctx.strokeStyle = 'rgba(60, 70, 80, 0.5)';
        ctx.lineWidth = 1;
      }
      ctx.stroke();

      // 선택/호버 하이라이트
      if (isHovered || isSelected) {
        ctx.beginPath();
        ctx.roundRect(tile.rx, tile.ry, tile.rw, tile.rh, r);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.fill();
      }

      // 라벨: 지역명
      ctx.fillStyle = '#eee';
      ctx.font = `bold ${Math.max(10, tile.rw / 8)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(tile.label, tile.rx + tile.rw / 2, tile.ry + 16);

      if (regionData) {
        // 가격 (억)
        const priceEok = (regionData.price / 10000).toFixed(1);
        ctx.fillStyle = '#fff';
        ctx.font = `bold ${Math.max(11, tile.rw / 7)}px monospace`;
        ctx.fillText(`${priceEok}억`, tile.rx + tile.rw / 2, tile.ry + tile.rh - 20);

        // 변화율
        const chg = regionData.price_change * 100;
        const chgStr = `${chg >= 0 ? '+' : ''}${chg.toFixed(1)}%`;
        ctx.fillStyle = chg > 0 ? '#ff6644' : chg < 0 ? '#44aaff' : '#888';
        ctx.font = `${Math.max(9, tile.rw / 9)}px monospace`;
        ctx.fillText(chgStr, tile.rx + tile.rw / 2, tile.ry + tile.rh - 7);

        // 에이전트 파티클
        const particles = sampleParticles(regionData, tile.rx, tile.ry, tile.rw, tile.rh);
        for (const p of particles) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
          ctx.fillStyle = p.color;
          ctx.fill();
        }
      } else {
        // 데이터 없을 때
        ctx.fillStyle = '#666';
        ctx.font = `${Math.max(9, tile.rw / 10)}px sans-serif`;
        ctx.fillText('대기', tile.rx + tile.rw / 2, tile.ry + tile.rh / 2 + 10);
      }
    }

    // 범례
    const legendY = canvasSize.h - 16;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    const legends = [
      { color: '#44dd66', label: '유주택' },
      { color: '#ffcc33', label: '무주택' },
      { color: '#ff6644', label: '다주택' },
    ];
    let lx = 8;
    for (const leg of legends) {
      ctx.fillStyle = leg.color;
      ctx.beginPath(); ctx.arc(lx + 4, legendY, 3, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#888';
      ctx.fillText(leg.label, lx + 10, legendY + 3);
      lx += 52;
    }
  }, [state, canvasSize, hoveredRegion, selectedRegion, getTileRects]);

  return (
    <div ref={containerRef} className="game-map-container">
      <canvas
        ref={canvasRef}
        className="game-map-canvas"
        style={{ width: canvasSize.w, height: canvasSize.h }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
      {/* 툴팁 */}
      {tooltip && (
        <div className="map-tooltip" style={{
          left: Math.min(tooltip.x + 12, canvasSize.w - 200),
          top: Math.min(tooltip.y + 12, canvasSize.h - 140),
        }}>
          <div className="tooltip-title">{tooltip.region.name}</div>
          <div className="tooltip-row">
            <span>가격</span><span>{(tooltip.region.price / 10000).toFixed(1)}억</span>
          </div>
          <div className="tooltip-row">
            <span>변화</span>
            <span style={{ color: tooltip.region.price_change > 0 ? '#ff6644' : '#44aaff' }}>
              {(tooltip.region.price_change * 100).toFixed(1)}%
            </span>
          </div>
          <div className="tooltip-row">
            <span>거래</span><span>{tooltip.region.transactions}건</span>
          </div>
          <div className="tooltip-row">
            <span>수요/공급</span>
            <span>{tooltip.region.demand}/{tooltip.region.supply}</span>
          </div>
          <div className="tooltip-divider" />
          <div className="tooltip-row">
            <span>무주택</span><span>{tooltip.region.homeless_count.toLocaleString()}</span>
          </div>
          <div className="tooltip-row">
            <span>1주택</span><span>{tooltip.region.one_house_count.toLocaleString()}</span>
          </div>
          <div className="tooltip-row">
            <span>다주택</span><span>{tooltip.region.multi_house_count.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
