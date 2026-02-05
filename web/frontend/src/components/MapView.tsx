/**
 * Leaflet 지도 컴포넌트 - 지역별 가격/거래 시각화
 */

import React, { useEffect, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { MonthlyState, RegionStats, REGION_COORDS, REGION_NAMES } from '../types/simulation';
import 'leaflet/dist/leaflet.css';

interface MapViewProps {
  state: MonthlyState | null;
  showAgentDist: boolean;
  showTransactions: boolean;
}

// 가격에 따른 색상 계산 (빨강: 고가, 파랑: 저가)
function getPriceColor(price: number, minPrice: number, maxPrice: number): string {
  if (maxPrice === minPrice) return '#ffff00';

  const ratio = (price - minPrice) / (maxPrice - minPrice);
  // 고가: 빨강, 저가: 파랑
  const r = Math.round(255 * ratio);
  const b = Math.round(255 * (1 - ratio));
  return `rgb(${r}, 50, ${b})`;
}

// 가격 변화에 따른 색상 (상승: 빨강, 하락: 파랑)
function getChangeColor(change: number): string {
  if (change > 0.02) return '#ff4444';
  if (change > 0.01) return '#ff8844';
  if (change > 0) return '#ffbb44';
  if (change > -0.01) return '#44ff44';
  if (change > -0.02) return '#44bbff';
  return '#4444ff';
}

// 에이전트 분포 색상 (무주택: 파랑, 1주택: 녹색, 다주택: 빨강)
function getAgentColor(type: 'homeless' | 'one_house' | 'multi_house'): string {
  switch (type) {
    case 'homeless': return '#4444ff';
    case 'one_house': return '#44bb44';
    case 'multi_house': return '#ff4444';
  }
}

// 지도 중심 및 줌 조정 컴포넌트
function MapController() {
  const map = useMap();

  useEffect(() => {
    // 한국 중심으로 설정
    map.setView([36.5, 127.5], 7);
  }, [map]);

  return null;
}

// 거래 펄스 애니메이션 마커
function TransactionMarker({ region, count, avgPrice }: {
  region: RegionStats;
  count: number;
  avgPrice: number;
}) {
  const coords = REGION_COORDS[region.region_id];
  if (!coords) return null;

  // 거래량에 따른 반경 (최소 5, 최대 30)
  const radius = Math.min(30, Math.max(5, Math.sqrt(count) * 3));

  return (
    <CircleMarker
      center={[coords.lat, coords.lng]}
      radius={radius}
      pathOptions={{
        color: '#ff6600',
        fillColor: '#ff9933',
        fillOpacity: 0.6,
        weight: 2,
      }}
    >
      <Popup>
        <div className="popup-content">
          <h4>{region.name}</h4>
          <p>거래: {count}건</p>
          <p>평균가: {(avgPrice / 10000).toFixed(1)}억</p>
        </div>
      </Popup>
    </CircleMarker>
  );
}

// 지역 마커
function RegionMarker({ region, minPrice, maxPrice, showAgentDist }: {
  region: RegionStats;
  minPrice: number;
  maxPrice: number;
  showAgentDist: boolean;
}) {
  const coords = REGION_COORDS[region.region_id];
  if (!coords) return null;

  const priceColor = getPriceColor(region.price, minPrice, maxPrice);
  const changeColor = getChangeColor(region.price_change);

  // 에이전트 분포 표시
  if (showAgentDist) {
    const total = region.homeless_count + region.one_house_count + region.multi_house_count;
    const homelessRatio = total > 0 ? region.homeless_count / total : 0;

    return (
      <CircleMarker
        center={[coords.lat, coords.lng]}
        radius={20}
        pathOptions={{
          color: getAgentColor('homeless'),
          fillColor: `rgba(68, 68, 255, ${homelessRatio})`,
          fillOpacity: 0.7,
          weight: 3,
        }}
      >
        <Popup>
          <div className="popup-content">
            <h4>{region.name}</h4>
            <p>무주택: {region.homeless_count.toLocaleString()}가구</p>
            <p>1주택: {region.one_house_count.toLocaleString()}가구</p>
            <p>다주택: {region.multi_house_count.toLocaleString()}가구</p>
          </div>
        </Popup>
      </CircleMarker>
    );
  }

  return (
    <CircleMarker
      center={[coords.lat, coords.lng]}
      radius={25}
      pathOptions={{
        color: changeColor,
        fillColor: priceColor,
        fillOpacity: 0.7,
        weight: 3,
      }}
    >
      <Popup>
        <div className="popup-content">
          <h4>{region.name}</h4>
          <p>가격: {(region.price / 10000).toFixed(1)}억</p>
          <p>변화: {(region.price_change * 100).toFixed(1)}%</p>
          <p>수요: {region.demand.toLocaleString()}</p>
          <p>공급: {region.supply.toLocaleString()}</p>
          <p>거래: {region.transactions.toLocaleString()}건</p>
        </div>
      </Popup>
    </CircleMarker>
  );
}

export function MapView({ state, showAgentDist, showTransactions }: MapViewProps) {
  // 가격 범위 계산
  const { minPrice, maxPrice } = useMemo(() => {
    if (!state || state.regions.length === 0) {
      return { minPrice: 20000, maxPrice: 200000 };
    }
    const prices = state.regions.map((r) => r.price);
    return {
      minPrice: Math.min(...prices),
      maxPrice: Math.max(...prices),
    };
  }, [state]);

  return (
    <div className="map-view">
      <div className="map-header">
        <h3>지역별 시장 현황</h3>
        {state && (
          <span className="map-date">
            {state.year}년 {(state.month % 12) + 1}월
          </span>
        )}
      </div>

      <MapContainer
        center={[36.5, 127.5]}
        zoom={7}
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
      >
        <MapController />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* 지역 마커 */}
        {state?.regions.map((region) => (
          <RegionMarker
            key={region.region_id}
            region={region}
            minPrice={minPrice}
            maxPrice={maxPrice}
            showAgentDist={showAgentDist}
          />
        ))}

        {/* 거래 마커 */}
        {showTransactions && state?.recent_transactions.map((trans, idx) => (
          <TransactionMarker
            key={`trans-${idx}`}
            region={state.regions.find((r) => r.region_id === trans.region_id) || state.regions[0]}
            count={trans.count}
            avgPrice={trans.avg_price}
          />
        ))}
      </MapContainer>

      {/* 범례 */}
      <div className="map-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ background: 'linear-gradient(to right, #4444ff, #ff4444)' }} />
          <span>가격 (저 → 고)</span>
        </div>
        <div className="legend-item">
          <div className="legend-circle" style={{ borderColor: '#ff4444' }} />
          <span>상승</span>
        </div>
        <div className="legend-item">
          <div className="legend-circle" style={{ borderColor: '#44bbff' }} />
          <span>하락</span>
        </div>
      </div>
    </div>
  );
}
