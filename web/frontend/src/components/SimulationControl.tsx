/**
 * 시뮬레이션 제어 컴포넌트
 */

import React from 'react';
import { SimulationStatus } from '../types/simulation';

interface SimulationControlProps {
  status: SimulationStatus;
  error: string | null;
}

export function SimulationControl({ status, error }: SimulationControlProps) {
  return (
    <div className="simulation-control">
      {/* 오류 표시 */}
      {error && (
        <div className="error-message">
          <span className="error-icon">⚠</span>
          <span>{error}</span>
        </div>
      )}

      {/* 로딩 표시 */}
      {(status === 'connecting' || status === 'initializing') && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <p>
            {status === 'connecting' ? '서버에 연결 중...' : '시뮬레이션 초기화 중...'}
          </p>
          <p className="loading-hint">
            100,000 에이전트 초기화에 약 10-30초가 소요됩니다.
          </p>
        </div>
      )}

      {/* 대기 상태 */}
      {status === 'idle' && (
        <div className="idle-message">
          <h3>시뮬레이션 대기 중</h3>
          <p>
            좌측 패널에서 파라미터를 설정하고 <strong>시뮬레이션 시작</strong> 버튼을 클릭하세요.
          </p>
          <div className="feature-list">
            <div className="feature-item">
              <span className="feature-icon">🏠</span>
              <span>100,000 에이전트 기반 시뮬레이션</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">📊</span>
              <span>행동경제학 기반 의사결정 모델</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">🗺</span>
              <span>13개 지역 실시간 시각화</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">⚙</span>
              <span>다양한 정책 시나리오 테스트</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
