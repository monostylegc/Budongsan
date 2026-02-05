/**
 * 파라미터 조절 패널 - 에이전트/환경 분리 + 분포 지원
 */

import React, { useState } from 'react';
import { SimulationParams, DistParam } from '../types/simulation';

interface ParameterPanelProps {
  onStart: (params: SimulationParams) => void;
  onStop: () => void;
  onReset: () => void;
  isRunning: boolean;
  isCompleted: boolean;
}

// 기본 파라미터
const defaultParams: SimulationParams = {
  num_households: 100000,
  num_houses: 60000,
  num_steps: 120,
  seed: 42,

  policy: {
    ltv_1house: 0.50, ltv_2house: 0.30, ltv_3house: 0.00,
    dti_limit: 0.40, dsr_limit: 0.40,
    acq_tax_1house: 0.01, acq_tax_2house: 0.08, acq_tax_3house: 0.12,
    transfer_tax_short: 0.70, transfer_tax_long: 0.40,
    transfer_tax_multi_short: 0.75, transfer_tax_multi_long: 0.60,
    jongbu_threshold_1house: 110000, jongbu_threshold_multi: 60000, jongbu_rate: 0.02,
    interest_rate: 0.035, mortgage_spread: 0.015,
    jeonse_loan_limit: 50000, rent_increase_cap: 0.05,
  },

  behavioral: {
    // 분포 파라미터
    fomo_sensitivity: { mean: 0.5, std: 0.15 },
    loss_aversion: { mean: 2.5, std: 0.35 },
    anchoring_strength: { mean: 0.5, std: 0.15 },
    herding_tendency: { mean: 0.4, std: 0.15 },
    risk_tolerance: { mean: 0.4, std: 0.15 },
    present_bias: { mean: 0.7, std: 0.1 },
    // 환경/임계값
    fomo_trigger_threshold: 0.05,
    anchoring_threshold: 0.1,
    herding_trigger: 0.03,
    social_learning_rate: 0.1,
    news_impact: 0.2,
  },

  agent_composition: {
    investor_ratio: 0.15, speculator_ratio: 0.05,
    speculator_risk_multiplier: 1.5, speculator_fomo_multiplier: 1.3,
    speculator_horizon_min: 6, speculator_horizon_max: 24,
    initial_homeless_rate: 0.45, initial_one_house_rate: 0.40, initial_multi_house_rate: 0.15,
    // 소득 분포 (로그정규: 중위값 300만원, sigma 0.6)
    income_median: 300, income_sigma: 0.6,
    // 자산 분포 (파레토: 중위값 5000만원, alpha 1.5)
    asset_median: 5000, asset_alpha: 1.5,
    // 연령 분포
    age_young_ratio: 0.45, age_middle_ratio: 0.43, age_senior_ratio: 0.12,
  },

  lifecycle: {
    marriage_urgency_age_start: 28, marriage_urgency_age_end: 35, newlywed_housing_pressure: 1.5,
    parenting_housing_pressure: 1.3,
    school_transition_age_start: 10, school_transition_age_end: 15, school_district_premium: 1.2,
    retirement_start_age: 55, downsizing_probability: 0.1,
  },

  network: {
    avg_neighbors: 10, rewiring_prob: 0.1,
    cascade_threshold: 0.3, cascade_multiplier: 2.0, self_weight: 0.6,
  },

  macro: {
    m2_growth: 0.08, gdp_growth_mean: 0.025, gdp_growth_volatility: 0.01,
    inflation_target: 0.02, income_gdp_beta: 0.8,
  },

  supply: {
    base_supply_rate: 0.001,
    elasticity_gangnam: 0.3, elasticity_seoul: 0.5, elasticity_gyeonggi: 1.5, elasticity_local: 2.0,
    redevelopment_base_prob: 0.001, redevelopment_age_threshold: 30, construction_period: 24,
  },

  depreciation: {
    depreciation_rate: 0.002, natural_demolition_threshold: 0.1, disaster_rate: 0.0001,
  },

  market: {
    price_sensitivity: 0.001, expectation_weight: 0.015, base_appreciation: 0.002,
    buy_threshold: 0.25, sell_threshold: 0.30, spillover_rate: 0.005,
  },

  scenario: 'default',
};

// 단일 슬라이더
function Slider({ label, value, min, max, step, unit, onChange, disabled, format, desc }: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (value: number) => void; disabled: boolean; format?: (v: number) => string; desc?: string;
}) {
  const displayValue = format ? format(value) : value.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0);
  return (
    <div className="slider-group">
      <div className="slider-header">
        <span className="slider-label" title={desc}>{label}</span>
        <span className="slider-value">{displayValue}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} disabled={disabled} />
    </div>
  );
}

// 분포 슬라이더 (평균 + 다양성)
function DistSlider({ label, dist, minMean, maxMean, stepMean, minStd, maxStd, stepStd, unit, onChange, disabled, desc }: {
  label: string; dist: DistParam;
  minMean: number; maxMean: number; stepMean: number;
  minStd: number; maxStd: number; stepStd: number;
  unit: string; onChange: (dist: DistParam) => void; disabled: boolean; desc?: string;
}) {
  const meanDisplay = dist.mean.toFixed(stepMean < 0.1 ? 2 : stepMean < 1 ? 1 : 0);
  const stdDisplay = dist.std.toFixed(stepStd < 0.1 ? 2 : stepStd < 1 ? 1 : 0);

  return (
    <div className="dist-slider-group">
      <div className="dist-header">
        <span className="dist-label" title={desc}>{label}</span>
        <span className="dist-value">{meanDisplay} ± {stdDisplay}{unit}</span>
      </div>
      <div className="dist-sliders">
        <div className="dist-slider-row">
          <span className="dist-sub-label">평균</span>
          <input type="range" min={minMean} max={maxMean} step={stepMean} value={dist.mean}
            onChange={(e) => onChange({ ...dist, mean: Number(e.target.value) })} disabled={disabled} />
        </div>
        <div className="dist-slider-row">
          <span className="dist-sub-label">다양성</span>
          <input type="range" min={minStd} max={maxStd} step={stepStd} value={dist.std}
            onChange={(e) => onChange({ ...dist, std: Number(e.target.value) })} disabled={disabled} />
        </div>
      </div>
      <div className="dist-preview">
        <span>범위: {Math.max(minMean, dist.mean - 2*dist.std).toFixed(stepMean < 1 ? 1 : 0)} ~ {Math.min(maxMean, dist.mean + 2*dist.std).toFixed(stepMean < 1 ? 1 : 0)}{unit}</span>
      </div>
    </div>
  );
}

type Category = 'agent' | 'env' | 'sim';
type AgentTab = 'traits' | 'composition' | 'lifecycle' | 'network';
type EnvTab = 'loan' | 'tax' | 'macro' | 'supply' | 'market';

export function ParameterPanel({ onStart, onStop, onReset, isRunning, isCompleted }: ParameterPanelProps) {
  const [params, setParams] = useState<SimulationParams>(defaultParams);
  const [category, setCategory] = useState<Category>('agent');
  const [agentTab, setAgentTab] = useState<AgentTab>('traits');
  const [envTab, setEnvTab] = useState<EnvTab>('loan');

  // 업데이트 헬퍼
  const update = (section: string, key: string, value: any) => {
    setParams(prev => ({ ...prev, [section]: { ...(prev as any)[section], [key]: value } }));
  };

  const updateDist = (key: string, dist: DistParam) => {
    setParams(prev => ({ ...prev, behavioral: { ...prev.behavioral, [key]: dist } }));
  };

  // 프리셋
  const applyPreset = (preset: string) => {
    let p = JSON.parse(JSON.stringify(defaultParams));
    switch (preset) {
      case 'homo_high_fomo':
        // 동질적 + 높은 FOMO
        p.behavioral.fomo_sensitivity = { mean: 0.8, std: 0.05 };
        p.behavioral.herding_tendency = { mean: 0.7, std: 0.05 };
        p.behavioral.loss_aversion = { mean: 2.5, std: 0.1 };
        break;
      case 'hetero_high_fomo':
        // 이질적 + 높은 FOMO
        p.behavioral.fomo_sensitivity = { mean: 0.7, std: 0.25 };
        p.behavioral.herding_tendency = { mean: 0.6, std: 0.25 };
        p.behavioral.loss_aversion = { mean: 2.5, std: 0.5 };
        break;
      case 'rational':
        // 합리적 에이전트
        p.behavioral.fomo_sensitivity = { mean: 0.2, std: 0.1 };
        p.behavioral.herding_tendency = { mean: 0.2, std: 0.1 };
        p.behavioral.loss_aversion = { mean: 1.5, std: 0.2 };
        p.behavioral.present_bias = { mean: 0.9, std: 0.05 };
        p.agent_composition.speculator_ratio = 0.02;
        break;
      case 'diverse':
        // 매우 다양한 에이전트
        p.behavioral.fomo_sensitivity = { mean: 0.5, std: 0.3 };
        p.behavioral.herding_tendency = { mean: 0.4, std: 0.25 };
        p.behavioral.loss_aversion = { mean: 2.5, std: 0.6 };
        p.behavioral.risk_tolerance = { mean: 0.4, std: 0.25 };
        p.behavioral.present_bias = { mean: 0.7, std: 0.15 };
        break;
      case 'korea_reality':
        // 한국 현실 (2024년 통계 기반)
        // 출처: 통계청 가계금융복지조사, 국토연구원 자료
        // 소득: 중위 월소득 약 350만원, 지니계수 0.33 반영
        p.agent_composition.income_median = 350;
        p.agent_composition.income_sigma = 0.55;
        // 자산: 중위 순자산 약 6000만원, 상위집중도 반영 (파레토 α=1.3)
        p.agent_composition.asset_median = 6000;
        p.agent_composition.asset_alpha = 1.3;
        // 주택보유: 자가 56%, 무주택 44% (2023년 기준)
        p.agent_composition.initial_homeless_rate = 0.44;
        p.agent_composition.initial_one_house_rate = 0.41;
        p.agent_composition.initial_multi_house_rate = 0.15;
        // 연령분포: 고령화 사회 반영
        p.agent_composition.age_young_ratio = 0.35;
        p.agent_composition.age_middle_ratio = 0.45;
        p.agent_composition.age_senior_ratio = 0.20;
        // 투자/투기자 비율: 갭투자 성행 반영
        p.agent_composition.investor_ratio = 0.18;
        p.agent_composition.speculator_ratio = 0.08;
        // 행동특성: 한국 부동산 시장 특성 반영
        p.behavioral.fomo_sensitivity = { mean: 0.6, std: 0.2 };  // 높은 FOMO
        p.behavioral.herding_tendency = { mean: 0.55, std: 0.2 }; // 강한 군집행동
        p.behavioral.loss_aversion = { mean: 2.5, std: 0.4 };     // Genesove & Mayer (2001)
        break;
      case 'tight_reg':
        p.policy.ltv_1house = 0.40; p.policy.ltv_2house = 0.20; p.policy.dti_limit = 0.35;
        p.policy.acq_tax_2house = 0.12; p.policy.jongbu_rate = 0.03;
        break;
      case 'loose_reg':
        p.policy.ltv_1house = 0.70; p.policy.ltv_2house = 0.50; p.policy.dti_limit = 0.50;
        p.policy.acq_tax_2house = 0.04; p.policy.jongbu_rate = 0.01;
        break;
      case 'supply_crisis':
        p.supply.base_supply_rate = 0.0003; p.supply.elasticity_gangnam = 0.1;
        break;
    }
    setParams(p);
  };

  return (
    <div className="parameter-panel">
      <h2>시뮬레이션 설정</h2>

      {/* 대분류 탭 */}
      <div className="category-tabs">
        <button className={category === 'agent' ? 'active' : ''} onClick={() => setCategory('agent')}>
          👤 에이전트
        </button>
        <button className={category === 'env' ? 'active' : ''} onClick={() => setCategory('env')}>
          🏛 환경
        </button>
        <button className={category === 'sim' ? 'active' : ''} onClick={() => setCategory('sim')}>
          ⚙ 설정
        </button>
      </div>

      {/* 에이전트 설정 */}
      {category === 'agent' && (
        <>
          <div className="preset-buttons">
            <button onClick={() => applyPreset('korea_reality')} disabled={isRunning} title="2024년 한국 통계 기반">🇰🇷 한국현실</button>
            <button onClick={() => applyPreset('homo_high_fomo')} disabled={isRunning} title="동질적 + 높은 FOMO">동질FOMO</button>
            <button onClick={() => applyPreset('hetero_high_fomo')} disabled={isRunning} title="이질적 + 높은 FOMO">이질FOMO</button>
            <button onClick={() => applyPreset('rational')} disabled={isRunning} title="합리적 에이전트">합리적</button>
            <button onClick={() => applyPreset('diverse')} disabled={isRunning} title="매우 다양한 특성">다양성↑</button>
          </div>

          <div className="tabs">
            <button className={agentTab === 'traits' ? 'active' : ''} onClick={() => setAgentTab('traits')}>심리특성</button>
            <button className={agentTab === 'composition' ? 'active' : ''} onClick={() => setAgentTab('composition')}>구성</button>
            <button className={agentTab === 'lifecycle' ? 'active' : ''} onClick={() => setAgentTab('lifecycle')}>생애주기</button>
            <button className={agentTab === 'network' ? 'active' : ''} onClick={() => setAgentTab('network')}>네트워크</button>
          </div>

          <div className="tab-content">
            {agentTab === 'traits' && (
              <div className="sliders">
                <div className="section-title">FOMO (놓칠 것에 대한 두려움)</div>
                <DistSlider label="FOMO 민감도" dist={params.behavioral.fomo_sensitivity}
                  minMean={0} maxMean={1} stepMean={0.05} minStd={0} maxStd={0.4} stepStd={0.05} unit=""
                  onChange={d => updateDist('fomo_sensitivity', d)} disabled={isRunning}
                  desc="높을수록 가격 상승 시 매수 욕구 증가" />
                <Slider label="발동 임계값" value={params.behavioral.fomo_trigger_threshold * 100}
                  min={1} max={15} step={1} unit="%"
                  onChange={v => update('behavioral', 'fomo_trigger_threshold', v/100)} disabled={isRunning}
                  desc="6개월간 이 비율 이상 상승 시 FOMO 발동" />

                <div className="section-title">손실 회피</div>
                <DistSlider label="손실 회피 계수" dist={params.behavioral.loss_aversion}
                  minMean={1} maxMean={4} stepMean={0.1} minStd={0.1} maxStd={1} stepStd={0.05} unit=""
                  onChange={d => updateDist('loss_aversion', d)} disabled={isRunning}
                  desc="2.5가 표준 (손실이 이득보다 2.5배 아픔)" />

                <div className="section-title">앵커링 (매입가 집착)</div>
                <DistSlider label="앵커링 강도" dist={params.behavioral.anchoring_strength}
                  minMean={0} maxMean={1} stepMean={0.05} minStd={0} maxStd={0.4} stepStd={0.05} unit=""
                  onChange={d => updateDist('anchoring_strength', d)} disabled={isRunning}
                  desc="높을수록 매입가에 집착" />
                <Slider label="발동 이익률" value={params.behavioral.anchoring_threshold * 100}
                  min={0} max={30} step={5} unit="%"
                  onChange={v => update('behavioral', 'anchoring_threshold', v/100)} disabled={isRunning}
                  desc="이 이익률 미만에서 매도 꺼림" />

                <div className="section-title">군집 행동</div>
                <DistSlider label="군집 성향" dist={params.behavioral.herding_tendency}
                  minMean={0} maxMean={1} stepMean={0.05} minStd={0} maxStd={0.4} stepStd={0.05} unit=""
                  onChange={d => updateDist('herding_tendency', d)} disabled={isRunning}
                  desc="높을수록 남 따라하기" />
                <Slider label="발동 비율" value={params.behavioral.herding_trigger * 100}
                  min={1} max={10} step={1} unit="%"
                  onChange={v => update('behavioral', 'herding_trigger', v/100)} disabled={isRunning}
                  desc="지역 내 이 비율 이상이 매수 시도 시 따라함" />

                <div className="section-title">위험 성향</div>
                <DistSlider label="위험 허용도" dist={params.behavioral.risk_tolerance}
                  minMean={0.1} maxMean={0.9} stepMean={0.05} minStd={0} maxStd={0.3} stepStd={0.05} unit=""
                  onChange={d => updateDist('risk_tolerance', d)} disabled={isRunning}
                  desc="높을수록 위험 추구" />

                <div className="section-title">시간 선호</div>
                <DistSlider label="현재 편향 (β)" dist={params.behavioral.present_bias}
                  minMean={0.5} maxMean={1} stepMean={0.05} minStd={0} maxStd={0.2} stepStd={0.02} unit=""
                  onChange={d => updateDist('present_bias', d)} disabled={isRunning}
                  desc="낮을수록 단기 이익 선호 (1=합리적)" />

                <div className="section-title">사회적 학습</div>
                <Slider label="학습 속도" value={params.behavioral.social_learning_rate}
                  min={0} max={0.5} step={0.05} unit=""
                  onChange={v => update('behavioral', 'social_learning_rate', v)} disabled={isRunning} />
                <Slider label="뉴스 영향도" value={params.behavioral.news_impact}
                  min={0} max={0.5} step={0.05} unit=""
                  onChange={v => update('behavioral', 'news_impact', v)} disabled={isRunning} />
              </div>
            )}

            {agentTab === 'composition' && (
              <div className="sliders">
                <div className="section-title">에이전트 유형 비율</div>
                <Slider label="투자자 (임대수익)" value={params.agent_composition.investor_ratio * 100}
                  min={0} max={40} step={5} unit="%"
                  onChange={v => update('agent_composition', 'investor_ratio', v/100)} disabled={isRunning} />
                <Slider label="투기자 (시세차익)" value={params.agent_composition.speculator_ratio * 100}
                  min={0} max={30} step={1} unit="%"
                  onChange={v => update('agent_composition', 'speculator_ratio', v/100)} disabled={isRunning} />
                <div className="info-text">실수요자 = {(100 - params.agent_composition.investor_ratio*100 - params.agent_composition.speculator_ratio*100).toFixed(0)}%</div>

                <div className="section-title">투기자 특성 배율</div>
                <Slider label="위험 허용도" value={params.agent_composition.speculator_risk_multiplier}
                  min={1} max={3} step={0.1} unit="x"
                  onChange={v => update('agent_composition', 'speculator_risk_multiplier', v)} disabled={isRunning} />
                <Slider label="FOMO 민감도" value={params.agent_composition.speculator_fomo_multiplier}
                  min={1} max={3} step={0.1} unit="x"
                  onChange={v => update('agent_composition', 'speculator_fomo_multiplier', v)} disabled={isRunning} />
                <Slider label="최소 보유 기간" value={params.agent_composition.speculator_horizon_min}
                  min={1} max={12} step={1} unit="월"
                  onChange={v => update('agent_composition', 'speculator_horizon_min', v)} disabled={isRunning} />
                <Slider label="최대 보유 기간" value={params.agent_composition.speculator_horizon_max}
                  min={6} max={48} step={6} unit="월"
                  onChange={v => update('agent_composition', 'speculator_horizon_max', v)} disabled={isRunning} />

                <div className="section-title">초기 주택 보유 분포</div>
                <Slider label="무주택 비율" value={params.agent_composition.initial_homeless_rate * 100}
                  min={20} max={70} step={5} unit="%"
                  onChange={v => update('agent_composition', 'initial_homeless_rate', v/100)} disabled={isRunning} />
                <Slider label="1주택 비율" value={params.agent_composition.initial_one_house_rate * 100}
                  min={20} max={60} step={5} unit="%"
                  onChange={v => update('agent_composition', 'initial_one_house_rate', v/100)} disabled={isRunning} />
                <Slider label="다주택 비율" value={params.agent_composition.initial_multi_house_rate * 100}
                  min={5} max={30} step={5} unit="%"
                  onChange={v => update('agent_composition', 'initial_multi_house_rate', v/100)} disabled={isRunning} />

                <div className="section-title">소득 분포 (로그정규)</div>
                <Slider label="중위 소득" value={params.agent_composition.income_median}
                  min={150} max={800} step={50} unit="만원/월"
                  onChange={v => update('agent_composition', 'income_median', v)} disabled={isRunning}
                  desc="가구 월소득 중위값" />
                <Slider label="소득 분산도 (σ)" value={params.agent_composition.income_sigma}
                  min={0.3} max={1.2} step={0.1} unit=""
                  onChange={v => update('agent_composition', 'income_sigma', v)} disabled={isRunning}
                  desc="높을수록 소득 불평등 증가" />
                <div className="info-text">
                  하위 10%: {Math.round(params.agent_composition.income_median * Math.exp(-1.28 * params.agent_composition.income_sigma))}만원 |
                  상위 10%: {Math.round(params.agent_composition.income_median * Math.exp(1.28 * params.agent_composition.income_sigma))}만원
                </div>

                <div className="section-title">자산 분포 (파레토)</div>
                <Slider label="중위 자산" value={params.agent_composition.asset_median / 10000}
                  min={0.3} max={3} step={0.1} unit="억"
                  onChange={v => update('agent_composition', 'asset_median', v * 10000)} disabled={isRunning}
                  desc="가구 순자산 중위값" />
                <Slider label="불평등도 (α)" value={params.agent_composition.asset_alpha}
                  min={1.1} max={3} step={0.1} unit=""
                  onChange={v => update('agent_composition', 'asset_alpha', v)} disabled={isRunning}
                  desc="낮을수록 상위 집중 (1.5=한국)" />
                <div className="info-text">
                  상위 10% 점유율: {(100 * Math.pow(0.1, 1 - 1/params.agent_composition.asset_alpha)).toFixed(0)}%
                </div>

                <div className="section-title">연령 분포</div>
                <Slider label="청년층 (25-34세)" value={params.agent_composition.age_young_ratio * 100}
                  min={20} max={60} step={5} unit="%"
                  onChange={v => update('agent_composition', 'age_young_ratio', v/100)} disabled={isRunning} />
                <Slider label="중년층 (35-54세)" value={params.agent_composition.age_middle_ratio * 100}
                  min={20} max={60} step={5} unit="%"
                  onChange={v => update('agent_composition', 'age_middle_ratio', v/100)} disabled={isRunning} />
                <Slider label="장년층 (55세+)" value={params.agent_composition.age_senior_ratio * 100}
                  min={5} max={40} step={5} unit="%"
                  onChange={v => update('agent_composition', 'age_senior_ratio', v/100)} disabled={isRunning} />
                <div className="info-text">
                  합계: {((params.agent_composition.age_young_ratio + params.agent_composition.age_middle_ratio + params.agent_composition.age_senior_ratio) * 100).toFixed(0)}%
                  {Math.abs(params.agent_composition.age_young_ratio + params.agent_composition.age_middle_ratio + params.agent_composition.age_senior_ratio - 1) > 0.01 &&
                    <span style={{color: '#e74c3c'}}> (100%가 되어야 함)</span>}
                </div>
              </div>
            )}

            {agentTab === 'lifecycle' && (
              <div className="sliders">
                <div className="section-title">결혼</div>
                <Slider label="주거 압박 시작 나이" value={params.lifecycle.marriage_urgency_age_start}
                  min={22} max={35} step={1} unit="세"
                  onChange={v => update('lifecycle', 'marriage_urgency_age_start', v)} disabled={isRunning} />
                <Slider label="주거 압박 종료 나이" value={params.lifecycle.marriage_urgency_age_end}
                  min={30} max={45} step={1} unit="세"
                  onChange={v => update('lifecycle', 'marriage_urgency_age_end', v)} disabled={isRunning} />
                <Slider label="신혼 압박 배율" value={params.lifecycle.newlywed_housing_pressure}
                  min={1} max={3} step={0.1} unit="x"
                  onChange={v => update('lifecycle', 'newlywed_housing_pressure', v)} disabled={isRunning} />

                <div className="section-title">육아</div>
                <Slider label="육아기 압박 배율" value={params.lifecycle.parenting_housing_pressure}
                  min={1} max={2.5} step={0.1} unit="x"
                  onChange={v => update('lifecycle', 'parenting_housing_pressure', v)} disabled={isRunning} />

                <div className="section-title">학군</div>
                <Slider label="이동 시작 자녀 나이" value={params.lifecycle.school_transition_age_start}
                  min={6} max={15} step={1} unit="세"
                  onChange={v => update('lifecycle', 'school_transition_age_start', v)} disabled={isRunning} />
                <Slider label="이동 종료 자녀 나이" value={params.lifecycle.school_transition_age_end}
                  min={12} max={18} step={1} unit="세"
                  onChange={v => update('lifecycle', 'school_transition_age_end', v)} disabled={isRunning} />
                <Slider label="학군 지역 선호 배율" value={params.lifecycle.school_district_premium}
                  min={1} max={2} step={0.1} unit="x"
                  onChange={v => update('lifecycle', 'school_district_premium', v)} disabled={isRunning} />

                <div className="section-title">은퇴</div>
                <Slider label="은퇴 고려 시작 나이" value={params.lifecycle.retirement_start_age}
                  min={50} max={65} step={1} unit="세"
                  onChange={v => update('lifecycle', 'retirement_start_age', v)} disabled={isRunning} />
                <Slider label="연간 다운사이징 확률" value={params.lifecycle.downsizing_probability * 100}
                  min={0} max={30} step={1} unit="%"
                  onChange={v => update('lifecycle', 'downsizing_probability', v/100)} disabled={isRunning} />
              </div>
            )}

            {agentTab === 'network' && (
              <div className="sliders">
                <div className="section-title">Small-World 네트워크</div>
                <Slider label="평균 이웃 수" value={params.network.avg_neighbors}
                  min={2} max={30} step={2} unit="명"
                  onChange={v => update('network', 'avg_neighbors', v)} disabled={isRunning}
                  desc="정보 공유하는 이웃 수" />
                <Slider label="재연결 확률" value={params.network.rewiring_prob}
                  min={0} max={0.5} step={0.05} unit=""
                  onChange={v => update('network', 'rewiring_prob', v)} disabled={isRunning}
                  desc="타지역과의 연결 비율" />

                <div className="section-title">정보 캐스케이드</div>
                <Slider label="발동 임계값" value={params.network.cascade_threshold * 100}
                  min={10} max={70} step={5} unit="%"
                  onChange={v => update('network', 'cascade_threshold', v/100)} disabled={isRunning}
                  desc="이웃 중 이 비율 이상 매수 시 캐스케이드" />
                <Slider label="배율" value={params.network.cascade_multiplier}
                  min={1} max={5} step={0.5} unit="x"
                  onChange={v => update('network', 'cascade_multiplier', v)} disabled={isRunning} />

                <div className="section-title">DeGroot 학습</div>
                <Slider label="자기 신호 가중치" value={params.network.self_weight}
                  min={0.3} max={0.9} step={0.1} unit=""
                  onChange={v => update('network', 'self_weight', v)} disabled={isRunning}
                  desc="나머지는 이웃 신호 가중치" />
              </div>
            )}
          </div>
        </>
      )}

      {/* 환경 설정 */}
      {category === 'env' && (
        <>
          <div className="preset-buttons">
            <button onClick={() => applyPreset('tight_reg')} disabled={isRunning}>규제 강화</button>
            <button onClick={() => applyPreset('loose_reg')} disabled={isRunning}>규제 완화</button>
            <button onClick={() => applyPreset('supply_crisis')} disabled={isRunning}>공급 부족</button>
          </div>

          <div className="tabs">
            <button className={envTab === 'loan' ? 'active' : ''} onClick={() => setEnvTab('loan')}>대출</button>
            <button className={envTab === 'tax' ? 'active' : ''} onClick={() => setEnvTab('tax')}>세금</button>
            <button className={envTab === 'macro' ? 'active' : ''} onClick={() => setEnvTab('macro')}>거시경제</button>
            <button className={envTab === 'supply' ? 'active' : ''} onClick={() => setEnvTab('supply')}>공급</button>
            <button className={envTab === 'market' ? 'active' : ''} onClick={() => setEnvTab('market')}>시장</button>
          </div>

          <div className="tab-content">
            {envTab === 'loan' && (
              <div className="sliders">
                <div className="section-title">LTV (주택담보대출비율)</div>
                <Slider label="1주택자" value={params.policy.ltv_1house * 100} min={20} max={80} step={5} unit="%" onChange={v => update('policy', 'ltv_1house', v/100)} disabled={isRunning} />
                <Slider label="2주택자" value={params.policy.ltv_2house * 100} min={0} max={60} step={5} unit="%" onChange={v => update('policy', 'ltv_2house', v/100)} disabled={isRunning} />
                <Slider label="3주택+" value={params.policy.ltv_3house * 100} min={0} max={40} step={5} unit="%" onChange={v => update('policy', 'ltv_3house', v/100)} disabled={isRunning} />

                <div className="section-title">DTI/DSR</div>
                <Slider label="DTI 한도" value={params.policy.dti_limit * 100} min={20} max={70} step={5} unit="%" onChange={v => update('policy', 'dti_limit', v/100)} disabled={isRunning} />
                <Slider label="DSR 한도" value={params.policy.dsr_limit * 100} min={20} max={70} step={5} unit="%" onChange={v => update('policy', 'dsr_limit', v/100)} disabled={isRunning} />

                <div className="section-title">금리</div>
                <Slider label="기준금리" value={params.policy.interest_rate * 100} min={1} max={8} step={0.25} unit="%" onChange={v => update('policy', 'interest_rate', v/100)} disabled={isRunning} />
                <Slider label="모기지 스프레드" value={params.policy.mortgage_spread * 100} min={0.5} max={3} step={0.25} unit="%p" onChange={v => update('policy', 'mortgage_spread', v/100)} disabled={isRunning} />
              </div>
            )}

            {envTab === 'tax' && (
              <div className="sliders">
                <div className="section-title">취득세</div>
                <Slider label="1주택자" value={params.policy.acq_tax_1house * 100} min={0.5} max={5} step={0.5} unit="%" onChange={v => update('policy', 'acq_tax_1house', v/100)} disabled={isRunning} />
                <Slider label="2주택자" value={params.policy.acq_tax_2house * 100} min={1} max={15} step={1} unit="%" onChange={v => update('policy', 'acq_tax_2house', v/100)} disabled={isRunning} />
                <Slider label="3주택+" value={params.policy.acq_tax_3house * 100} min={1} max={20} step={1} unit="%" onChange={v => update('policy', 'acq_tax_3house', v/100)} disabled={isRunning} />

                <div className="section-title">양도세</div>
                <Slider label="단기 (2년-)" value={params.policy.transfer_tax_short * 100} min={30} max={80} step={5} unit="%" onChange={v => update('policy', 'transfer_tax_short', v/100)} disabled={isRunning} />
                <Slider label="장기 (2년+)" value={params.policy.transfer_tax_long * 100} min={10} max={60} step={5} unit="%" onChange={v => update('policy', 'transfer_tax_long', v/100)} disabled={isRunning} />
                <Slider label="다주택 단기" value={params.policy.transfer_tax_multi_short * 100} min={40} max={90} step={5} unit="%" onChange={v => update('policy', 'transfer_tax_multi_short', v/100)} disabled={isRunning} />
                <Slider label="다주택 장기" value={params.policy.transfer_tax_multi_long * 100} min={20} max={80} step={5} unit="%" onChange={v => update('policy', 'transfer_tax_multi_long', v/100)} disabled={isRunning} />

                <div className="section-title">종합부동산세</div>
                <Slider label="세율" value={params.policy.jongbu_rate * 100} min={0.5} max={5} step={0.5} unit="%" onChange={v => update('policy', 'jongbu_rate', v/100)} disabled={isRunning} />
                <Slider label="1주택 기준" value={params.policy.jongbu_threshold_1house / 10000} min={6} max={20} step={1} unit="억" onChange={v => update('policy', 'jongbu_threshold_1house', v*10000)} disabled={isRunning} />
                <Slider label="다주택 기준" value={params.policy.jongbu_threshold_multi / 10000} min={3} max={12} step={1} unit="억" onChange={v => update('policy', 'jongbu_threshold_multi', v*10000)} disabled={isRunning} />

                <div className="section-title">전월세</div>
                <Slider label="전세대출 한도" value={params.policy.jeonse_loan_limit / 10000} min={2} max={10} step={0.5} unit="억" onChange={v => update('policy', 'jeonse_loan_limit', v*10000)} disabled={isRunning} />
                <Slider label="임대료 상한" value={params.policy.rent_increase_cap * 100} min={0} max={10} step={1} unit="%" onChange={v => update('policy', 'rent_increase_cap', v/100)} disabled={isRunning} />
              </div>
            )}

            {envTab === 'macro' && (
              <div className="sliders">
                <div className="section-title">통화/경제 성장</div>
                <Slider label="M2 증가율 (연)" value={params.macro.m2_growth * 100} min={2} max={25} step={1} unit="%" onChange={v => update('macro', 'm2_growth', v/100)} disabled={isRunning} />
                <Slider label="GDP 성장률 (연)" value={params.macro.gdp_growth_mean * 100} min={-3} max={10} step={0.5} unit="%" onChange={v => update('macro', 'gdp_growth_mean', v/100)} disabled={isRunning} />
                <Slider label="GDP 변동성" value={params.macro.gdp_growth_volatility * 100} min={0} max={5} step={0.5} unit="%" onChange={v => update('macro', 'gdp_growth_volatility', v/100)} disabled={isRunning} />
                <Slider label="인플레 목표" value={params.macro.inflation_target * 100} min={0} max={5} step={0.5} unit="%" onChange={v => update('macro', 'inflation_target', v/100)} disabled={isRunning} />
                <Slider label="소득-GDP 탄력성" value={params.macro.income_gdp_beta} min={0.3} max={1.5} step={0.1} unit="" onChange={v => update('macro', 'income_gdp_beta', v)} disabled={isRunning} />
              </div>
            )}

            {envTab === 'supply' && (
              <div className="sliders">
                <div className="section-title">신규 공급</div>
                <Slider label="기본 공급률 (월)" value={params.supply.base_supply_rate * 1000} min={0.1} max={5} step={0.1} unit="‰" onChange={v => update('supply', 'base_supply_rate', v/1000)} disabled={isRunning} />

                <div className="section-title">공급 탄력성</div>
                <Slider label="강남" value={params.supply.elasticity_gangnam} min={0.05} max={1} step={0.05} unit="" onChange={v => update('supply', 'elasticity_gangnam', v)} disabled={isRunning} />
                <Slider label="기타 서울" value={params.supply.elasticity_seoul} min={0.1} max={1.5} step={0.1} unit="" onChange={v => update('supply', 'elasticity_seoul', v)} disabled={isRunning} />
                <Slider label="경기" value={params.supply.elasticity_gyeonggi} min={0.5} max={3} step={0.1} unit="" onChange={v => update('supply', 'elasticity_gyeonggi', v)} disabled={isRunning} />
                <Slider label="지방" value={params.supply.elasticity_local} min={1} max={5} step={0.5} unit="" onChange={v => update('supply', 'elasticity_local', v)} disabled={isRunning} />

                <div className="section-title">재건축</div>
                <Slider label="기본 확률 (월)" value={params.supply.redevelopment_base_prob * 1000} min={0} max={5} step={0.5} unit="‰" onChange={v => update('supply', 'redevelopment_base_prob', v/1000)} disabled={isRunning} />
                <Slider label="가능 연식" value={params.supply.redevelopment_age_threshold} min={20} max={50} step={5} unit="년" onChange={v => update('supply', 'redevelopment_age_threshold', v)} disabled={isRunning} />
                <Slider label="건설 기간" value={params.supply.construction_period} min={12} max={48} step={6} unit="월" onChange={v => update('supply', 'construction_period', v)} disabled={isRunning} />

                <div className="section-title">노후화/멸실</div>
                <Slider label="노후화율 (월)" value={params.depreciation.depreciation_rate * 1000} min={0.5} max={5} step={0.5} unit="‰" onChange={v => update('depreciation', 'depreciation_rate', v/1000)} disabled={isRunning} />
                <Slider label="자연 멸실 임계값" value={params.depreciation.natural_demolition_threshold * 100} min={5} max={30} step={5} unit="%" onChange={v => update('depreciation', 'natural_demolition_threshold', v/100)} disabled={isRunning} />
                <Slider label="재해 멸실률 (월)" value={params.depreciation.disaster_rate * 10000} min={0} max={10} step={1} unit="‱" onChange={v => update('depreciation', 'disaster_rate', v/10000)} disabled={isRunning} />
              </div>
            )}

            {envTab === 'market' && (
              <div className="sliders">
                <div className="section-title">가격 결정</div>
                <Slider label="수요/공급 민감도" value={params.market.price_sensitivity * 10000} min={1} max={20} step={1} unit="‱" onChange={v => update('market', 'price_sensitivity', v/10000)} disabled={isRunning} />
                <Slider label="기대 가중치" value={params.market.expectation_weight * 1000} min={1} max={30} step={1} unit="‰" onChange={v => update('market', 'expectation_weight', v/1000)} disabled={isRunning} />
                <Slider label="기본 상승률 (월)" value={params.market.base_appreciation * 100} min={0} max={1} step={0.1} unit="%" onChange={v => update('market', 'base_appreciation', v/100)} disabled={isRunning} />

                <div className="section-title">의사결정 임계값</div>
                <Slider label="매수 임계값" value={params.market.buy_threshold} min={0.1} max={0.5} step={0.05} unit="" onChange={v => update('market', 'buy_threshold', v)} disabled={isRunning} />
                <Slider label="매도 임계값" value={params.market.sell_threshold} min={0.1} max={0.6} step={0.05} unit="" onChange={v => update('market', 'sell_threshold', v)} disabled={isRunning} />

                <div className="section-title">풍선효과</div>
                <Slider label="전파 속도" value={params.market.spillover_rate * 1000} min={1} max={30} step={1} unit="‰" onChange={v => update('market', 'spillover_rate', v/1000)} disabled={isRunning} />
              </div>
            )}
          </div>
        </>
      )}

      {/* 시뮬레이션 설정 */}
      {category === 'sim' && (
        <div className="tab-content">
          <div className="sliders">
            <div className="section-title">규모</div>
            <Slider label="가구 수" value={params.num_households / 1000} min={10} max={500} step={10} unit="천"
              onChange={v => setParams(p => ({...p, num_households: v*1000}))} disabled={isRunning} format={v => v.toFixed(0)} />
            <Slider label="주택 수" value={params.num_houses / 1000} min={6} max={300} step={6} unit="천"
              onChange={v => setParams(p => ({...p, num_houses: v*1000}))} disabled={isRunning} format={v => v.toFixed(0)} />

            <div className="section-title">기간</div>
            <Slider label="시뮬레이션 기간" value={params.num_steps} min={12} max={360} step={12} unit="월"
              onChange={v => setParams(p => ({...p, num_steps: v}))} disabled={isRunning} />
            <div className="info-text">= {(params.num_steps / 12).toFixed(0)}년</div>

            <div className="section-title">기타</div>
            <Slider label="랜덤 시드" value={params.seed} min={1} max={1000} step={1} unit=""
              onChange={v => setParams(p => ({...p, seed: v}))} disabled={isRunning} />
          </div>
        </div>
      )}

      {/* 현재 설정 요약 */}
      <div className="settings-summary">
        <div className="summary-section">
          <span className="summary-title">👤 에이전트 (평균±다양성)</span>
          <div className="summary-row"><span>FOMO</span><span>{params.behavioral.fomo_sensitivity.mean.toFixed(1)}±{params.behavioral.fomo_sensitivity.std.toFixed(2)}</span></div>
          <div className="summary-row"><span>손실회피</span><span>{params.behavioral.loss_aversion.mean.toFixed(1)}±{params.behavioral.loss_aversion.std.toFixed(2)}</span></div>
          <div className="summary-row"><span>투기자</span><span>{(params.agent_composition.speculator_ratio*100).toFixed(0)}%</span></div>
        </div>
        <div className="summary-section">
          <span className="summary-title">🏛 환경</span>
          <div className="summary-row"><span>LTV</span><span>{(params.policy.ltv_1house*100).toFixed(0)}% / {(params.policy.ltv_2house*100).toFixed(0)}%</span></div>
          <div className="summary-row"><span>금리</span><span>{(params.policy.interest_rate*100).toFixed(2)}%</span></div>
        </div>
        <div className="summary-section">
          <span className="summary-title">⚙ 설정</span>
          <div className="summary-row"><span>규모</span><span>{(params.num_households/1000).toFixed(0)}K / {params.num_steps}월</span></div>
        </div>
      </div>

      {/* 컨트롤 버튼 */}
      <div className="controls">
        {!isRunning && !isCompleted && (
          <button className="btn-start" onClick={() => onStart(params)}>시뮬레이션 시작</button>
        )}
        {isRunning && (
          <button className="btn-stop" onClick={onStop}>중지</button>
        )}
        {(isCompleted || isRunning) && (
          <button className="btn-reset" onClick={onReset}>리셋</button>
        )}
      </div>
    </div>
  );
}
