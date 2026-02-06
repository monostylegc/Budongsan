/**
 * 설정 페이지 - 프리셋 카드 + 핵심 슬라이더 + 접기식 고급 설정
 */

import React, { useState } from 'react';
import { SimulationParams, DistParam } from '../types/simulation';

interface SetupPageProps {
  onStart: (params: SimulationParams) => void;
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
    fomo_sensitivity: { mean: 0.5, std: 0.15 },
    loss_aversion: { mean: 2.5, std: 0.35 },
    anchoring_strength: { mean: 0.5, std: 0.15 },
    herding_tendency: { mean: 0.4, std: 0.15 },
    risk_tolerance: { mean: 0.4, std: 0.15 },
    present_bias: { mean: 0.7, std: 0.1 },
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
    income_median: 300, income_sigma: 0.6,
    asset_median: 5000, asset_alpha: 1.5,
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

// 프리셋 정의
interface Preset {
  id: string;
  name: string;
  emoji: string;
  desc: string;
  color: string;
  apply: (p: SimulationParams) => SimulationParams;
}

const PRESETS: Preset[] = [
  {
    id: 'korea_2026', name: '한국 현실 2026', emoji: '🇰🇷', desc: '기준금리 2.5%, GDP 1.5%, 현행 규제',
    color: '#2196F3',
    apply: (p) => {
      const n = JSON.parse(JSON.stringify(p));
      n.num_households = 30000; n.num_houses = 24600; n.num_steps = 60;
      n.policy.interest_rate = 0.025; n.policy.mortgage_spread = 0.01;
      n.policy.ltv_1house = 0.40; n.policy.ltv_2house = 0.00; n.policy.ltv_3house = 0.00;
      n.policy.dti_limit = 0.40; n.policy.dsr_limit = 0.40;
      n.macro.gdp_growth_mean = 0.015; n.macro.gdp_growth_volatility = 0.01;
      n.agent_composition.income_median = 350; n.agent_composition.income_sigma = 0.55;
      n.agent_composition.asset_median = 6000; n.agent_composition.asset_alpha = 1.3;
      n.agent_composition.initial_homeless_rate = 0.44;
      n.agent_composition.initial_one_house_rate = 0.41;
      n.agent_composition.initial_multi_house_rate = 0.15;
      n.agent_composition.investor_ratio = 0.18; n.agent_composition.speculator_ratio = 0.08;
      n.behavioral.fomo_sensitivity = { mean: 0.6, std: 0.2 };
      n.behavioral.herding_tendency = { mean: 0.55, std: 0.2 };
      return n;
    },
  },
  {
    id: 'recession', name: '경기 침체', emoji: '📉', desc: 'GDP -1%, 실업률 상승, 변동성 증가',
    color: '#FF5722',
    apply: (p) => {
      const n = JSON.parse(JSON.stringify(p));
      n.num_households = 30000; n.num_houses = 24600; n.num_steps = 60;
      n.policy.interest_rate = 0.025;
      n.macro.gdp_growth_mean = -0.01; n.macro.gdp_growth_volatility = 0.015;
      return n;
    },
  },
  {
    id: 'deregulation', name: '규제 완화', emoji: '🔓', desc: 'LTV 60%, 2주택 30%, 종부세 인하',
    color: '#4CAF50',
    apply: (p) => {
      const n = JSON.parse(JSON.stringify(p));
      n.num_households = 30000; n.num_houses = 24600; n.num_steps = 60;
      n.policy.interest_rate = 0.025;
      n.policy.ltv_1house = 0.60; n.policy.ltv_2house = 0.30;
      n.policy.dti_limit = 0.50; n.policy.jongbu_rate = 0.01;
      n.policy.transfer_tax_multi_long = 0.40;
      return n;
    },
  },
  {
    id: 'rate_cut', name: '금리 인하', emoji: '💰', desc: '기준금리 1.5%, 주담대 2.5%',
    color: '#9C27B0',
    apply: (p) => {
      const n = JSON.parse(JSON.stringify(p));
      n.num_households = 30000; n.num_houses = 24600; n.num_steps = 60;
      n.policy.interest_rate = 0.015; n.policy.mortgage_spread = 0.01;
      return n;
    },
  },
  {
    id: 'supply_cliff', name: '공급 절벽', emoji: '🏗', desc: '공급률 50% 감소, 강남 공급 극히 제한',
    color: '#FF9800',
    apply: (p) => {
      const n = JSON.parse(JSON.stringify(p));
      n.num_households = 30000; n.num_houses = 24600; n.num_steps = 60;
      n.policy.interest_rate = 0.025;
      n.supply.base_supply_rate = 0.0005;
      n.supply.elasticity_gangnam = 0.15; n.supply.elasticity_seoul = 0.25;
      n.supply.elasticity_gyeonggi = 0.8; n.supply.redevelopment_base_prob = 0.0003;
      return n;
    },
  },
];

// 슬라이더 컴포넌트
function Slider({ label, value, min, max, step, unit, onChange, format, desc }: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (value: number) => void; format?: (v: number) => string; desc?: string;
}) {
  const displayValue = format ? format(value) : value.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0);
  return (
    <div className="slider-group">
      <div className="slider-header">
        <span className="slider-label" title={desc}>{label}</span>
        <span className="slider-value">{displayValue}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

// 분포 슬라이더
function DistSlider({ label, dist, minMean, maxMean, stepMean, minStd, maxStd, stepStd, unit, onChange, desc }: {
  label: string; dist: DistParam;
  minMean: number; maxMean: number; stepMean: number;
  minStd: number; maxStd: number; stepStd: number;
  unit: string; onChange: (dist: DistParam) => void; desc?: string;
}) {
  return (
    <div className="dist-slider-group">
      <div className="dist-header">
        <span className="dist-label" title={desc}>{label}</span>
        <span className="dist-value">{dist.mean.toFixed(2)} ± {dist.std.toFixed(2)}{unit}</span>
      </div>
      <div className="dist-sliders">
        <div className="dist-slider-row">
          <span className="dist-sub-label">평균</span>
          <input type="range" min={minMean} max={maxMean} step={stepMean} value={dist.mean}
            onChange={(e) => onChange({ ...dist, mean: Number(e.target.value) })} />
        </div>
        <div className="dist-slider-row">
          <span className="dist-sub-label">다양성</span>
          <input type="range" min={minStd} max={maxStd} step={stepStd} value={dist.std}
            onChange={(e) => onChange({ ...dist, std: Number(e.target.value) })} />
        </div>
      </div>
    </div>
  );
}

type AdvancedTab = 'agent_traits' | 'agent_comp' | 'lifecycle' | 'network' | 'loan' | 'tax' | 'macro' | 'supply' | 'market';

export function SetupPage({ onStart }: SetupPageProps) {
  const [params, setParams] = useState<SimulationParams>(JSON.parse(JSON.stringify(defaultParams)));
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advancedTab, setAdvancedTab] = useState<AdvancedTab>('agent_traits');

  // 업데이트 헬퍼
  const update = (section: string, key: string, value: any) => {
    setParams(prev => ({ ...prev, [section]: { ...(prev as any)[section], [key]: value } }));
  };
  const updateDist = (key: string, dist: DistParam) => {
    setParams(prev => ({ ...prev, behavioral: { ...prev.behavioral, [key]: dist } }));
  };

  // 프리셋 적용
  const applyPreset = (preset: Preset) => {
    const newParams = preset.apply(JSON.parse(JSON.stringify(defaultParams)));
    setParams(newParams);
    setSelectedPreset(preset.id);
  };

  return (
    <div className="setup-page">
      {/* 헤더 */}
      <header className="setup-header">
        <h1>한국 부동산 ABM 시뮬레이션</h1>
        <p className="setup-subtitle">행동경제학 기반 Agent-Based Model</p>
      </header>

      <div className="setup-content">
        {/* 프리셋 카드 */}
        <section className="preset-section">
          <h2>시나리오 선택</h2>
          <div className="preset-cards">
            {PRESETS.map(preset => (
              <button
                key={preset.id}
                className={`preset-card ${selectedPreset === preset.id ? 'selected' : ''}`}
                style={{ '--accent': preset.color } as React.CSSProperties}
                onClick={() => applyPreset(preset)}
              >
                <span className="preset-emoji">{preset.emoji}</span>
                <span className="preset-name">{preset.name}</span>
                <span className="preset-desc">{preset.desc}</span>
              </button>
            ))}
          </div>
        </section>

        {/* 핵심 파라미터 슬라이더 */}
        <section className="core-params">
          <h2>핵심 파라미터</h2>
          <div className="core-sliders">
            <Slider label="가구 수" value={params.num_households / 1000}
              min={10} max={200} step={10} unit="천"
              onChange={v => setParams(p => ({ ...p, num_households: v * 1000 }))}
              format={v => v.toFixed(0)} />
            <Slider label="시뮬레이션 기간" value={params.num_steps}
              min={12} max={360} step={12} unit={`월 (${(params.num_steps / 12).toFixed(0)}년)`}
              onChange={v => setParams(p => ({ ...p, num_steps: v }))} />
            <Slider label="기준금리" value={params.policy.interest_rate * 100}
              min={0.5} max={8} step={0.25} unit="%"
              onChange={v => update('policy', 'interest_rate', v / 100)} />
            <Slider label="GDP 성장률" value={params.macro.gdp_growth_mean * 100}
              min={-3} max={10} step={0.5} unit="%"
              onChange={v => update('macro', 'gdp_growth_mean', v / 100)} />
            <Slider label="LTV (1주택)" value={params.policy.ltv_1house * 100}
              min={0} max={80} step={5} unit="%"
              onChange={v => update('policy', 'ltv_1house', v / 100)} />
            <Slider label="FOMO 민감도" value={params.behavioral.fomo_sensitivity.mean}
              min={0} max={1} step={0.05} unit=""
              onChange={v => updateDist('fomo_sensitivity', { ...params.behavioral.fomo_sensitivity, mean: v })} />
            <Slider label="투기자 비율" value={params.agent_composition.speculator_ratio * 100}
              min={0} max={30} step={1} unit="%"
              onChange={v => update('agent_composition', 'speculator_ratio', v / 100)} />
          </div>
        </section>

        {/* 고급 설정 (접기식) */}
        <section className="advanced-section">
          <button className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
            <span className={`toggle-arrow ${showAdvanced ? 'open' : ''}`}>&#9654;</span>
            고급 설정
          </button>

          {showAdvanced && (
            <div className="advanced-content">
              <div className="advanced-tabs">
                {([
                  ['agent_traits', '심리특성'], ['agent_comp', '구성'], ['lifecycle', '생애주기'],
                  ['network', '네트워크'], ['loan', '대출'], ['tax', '세금'],
                  ['macro', '거시경제'], ['supply', '공급'], ['market', '시장'],
                ] as [AdvancedTab, string][]).map(([key, label]) => (
                  <button key={key} className={advancedTab === key ? 'active' : ''}
                    onClick={() => setAdvancedTab(key)}>{label}</button>
                ))}
              </div>

              <div className="advanced-panel">
                {advancedTab === 'agent_traits' && (
                  <div className="sliders">
                    <div className="section-title">FOMO</div>
                    <DistSlider label="FOMO 민감도" dist={params.behavioral.fomo_sensitivity}
                      minMean={0} maxMean={1} stepMean={0.05} minStd={0} maxStd={0.4} stepStd={0.05} unit=""
                      onChange={d => updateDist('fomo_sensitivity', d)} desc="높을수록 가격 상승 시 매수 욕구" />
                    <Slider label="발동 임계값" value={params.behavioral.fomo_trigger_threshold * 100}
                      min={1} max={15} step={1} unit="%"
                      onChange={v => update('behavioral', 'fomo_trigger_threshold', v / 100)} />

                    <div className="section-title">손실 회피</div>
                    <DistSlider label="손실 회피 계수" dist={params.behavioral.loss_aversion}
                      minMean={1} maxMean={4} stepMean={0.1} minStd={0.1} maxStd={1} stepStd={0.05} unit=""
                      onChange={d => updateDist('loss_aversion', d)} />

                    <div className="section-title">앵커링</div>
                    <DistSlider label="앵커링 강도" dist={params.behavioral.anchoring_strength}
                      minMean={0} maxMean={1} stepMean={0.05} minStd={0} maxStd={0.4} stepStd={0.05} unit=""
                      onChange={d => updateDist('anchoring_strength', d)} />
                    <Slider label="발동 이익률" value={params.behavioral.anchoring_threshold * 100}
                      min={0} max={30} step={5} unit="%"
                      onChange={v => update('behavioral', 'anchoring_threshold', v / 100)} />

                    <div className="section-title">군집 행동</div>
                    <DistSlider label="군집 성향" dist={params.behavioral.herding_tendency}
                      minMean={0} maxMean={1} stepMean={0.05} minStd={0} maxStd={0.4} stepStd={0.05} unit=""
                      onChange={d => updateDist('herding_tendency', d)} />
                    <Slider label="발동 비율" value={params.behavioral.herding_trigger * 100}
                      min={1} max={10} step={1} unit="%"
                      onChange={v => update('behavioral', 'herding_trigger', v / 100)} />

                    <div className="section-title">위험/시간 선호</div>
                    <DistSlider label="위험 허용도" dist={params.behavioral.risk_tolerance}
                      minMean={0.1} maxMean={0.9} stepMean={0.05} minStd={0} maxStd={0.3} stepStd={0.05} unit=""
                      onChange={d => updateDist('risk_tolerance', d)} />
                    <DistSlider label="현재 편향" dist={params.behavioral.present_bias}
                      minMean={0.5} maxMean={1} stepMean={0.05} minStd={0} maxStd={0.2} stepStd={0.02} unit=""
                      onChange={d => updateDist('present_bias', d)} />

                    <div className="section-title">사회적 학습</div>
                    <Slider label="학습 속도" value={params.behavioral.social_learning_rate}
                      min={0} max={0.5} step={0.05} unit=""
                      onChange={v => update('behavioral', 'social_learning_rate', v)} />
                    <Slider label="뉴스 영향도" value={params.behavioral.news_impact}
                      min={0} max={0.5} step={0.05} unit=""
                      onChange={v => update('behavioral', 'news_impact', v)} />
                  </div>
                )}

                {advancedTab === 'agent_comp' && (
                  <div className="sliders">
                    <div className="section-title">유형 비율</div>
                    <Slider label="투자자 (임대수익)" value={params.agent_composition.investor_ratio * 100}
                      min={0} max={40} step={5} unit="%" onChange={v => update('agent_composition', 'investor_ratio', v / 100)} />
                    <Slider label="투기자 (시세차익)" value={params.agent_composition.speculator_ratio * 100}
                      min={0} max={30} step={1} unit="%" onChange={v => update('agent_composition', 'speculator_ratio', v / 100)} />

                    <div className="section-title">초기 주택 보유</div>
                    <Slider label="무주택 비율" value={params.agent_composition.initial_homeless_rate * 100}
                      min={20} max={70} step={5} unit="%" onChange={v => update('agent_composition', 'initial_homeless_rate', v / 100)} />
                    <Slider label="1주택 비율" value={params.agent_composition.initial_one_house_rate * 100}
                      min={20} max={60} step={5} unit="%" onChange={v => update('agent_composition', 'initial_one_house_rate', v / 100)} />
                    <Slider label="다주택 비율" value={params.agent_composition.initial_multi_house_rate * 100}
                      min={5} max={30} step={5} unit="%" onChange={v => update('agent_composition', 'initial_multi_house_rate', v / 100)} />

                    <div className="section-title">소득 분포</div>
                    <Slider label="중위 소득" value={params.agent_composition.income_median}
                      min={150} max={800} step={50} unit="만원/월" onChange={v => update('agent_composition', 'income_median', v)} />
                    <Slider label="소득 분산도" value={params.agent_composition.income_sigma}
                      min={0.3} max={1.2} step={0.1} unit="" onChange={v => update('agent_composition', 'income_sigma', v)} />

                    <div className="section-title">자산 분포</div>
                    <Slider label="중위 자산" value={params.agent_composition.asset_median / 10000}
                      min={0.3} max={3} step={0.1} unit="억" onChange={v => update('agent_composition', 'asset_median', v * 10000)} />
                    <Slider label="불평등도" value={params.agent_composition.asset_alpha}
                      min={1.1} max={3} step={0.1} unit="" onChange={v => update('agent_composition', 'asset_alpha', v)} />

                    <div className="section-title">연령 분포</div>
                    <Slider label="청년 (25-34)" value={params.agent_composition.age_young_ratio * 100}
                      min={20} max={60} step={5} unit="%" onChange={v => update('agent_composition', 'age_young_ratio', v / 100)} />
                    <Slider label="중년 (35-54)" value={params.agent_composition.age_middle_ratio * 100}
                      min={20} max={60} step={5} unit="%" onChange={v => update('agent_composition', 'age_middle_ratio', v / 100)} />
                    <Slider label="장년 (55+)" value={params.agent_composition.age_senior_ratio * 100}
                      min={5} max={40} step={5} unit="%" onChange={v => update('agent_composition', 'age_senior_ratio', v / 100)} />
                  </div>
                )}

                {advancedTab === 'lifecycle' && (
                  <div className="sliders">
                    <div className="section-title">결혼</div>
                    <Slider label="압박 시작 나이" value={params.lifecycle.marriage_urgency_age_start}
                      min={22} max={35} step={1} unit="세" onChange={v => update('lifecycle', 'marriage_urgency_age_start', v)} />
                    <Slider label="압박 종료 나이" value={params.lifecycle.marriage_urgency_age_end}
                      min={30} max={45} step={1} unit="세" onChange={v => update('lifecycle', 'marriage_urgency_age_end', v)} />
                    <Slider label="신혼 압박 배율" value={params.lifecycle.newlywed_housing_pressure}
                      min={1} max={3} step={0.1} unit="x" onChange={v => update('lifecycle', 'newlywed_housing_pressure', v)} />

                    <div className="section-title">학군</div>
                    <Slider label="이동 시작 자녀 나이" value={params.lifecycle.school_transition_age_start}
                      min={6} max={15} step={1} unit="세" onChange={v => update('lifecycle', 'school_transition_age_start', v)} />
                    <Slider label="학군 선호 배율" value={params.lifecycle.school_district_premium}
                      min={1} max={2} step={0.1} unit="x" onChange={v => update('lifecycle', 'school_district_premium', v)} />

                    <div className="section-title">은퇴</div>
                    <Slider label="은퇴 시작 나이" value={params.lifecycle.retirement_start_age}
                      min={50} max={65} step={1} unit="세" onChange={v => update('lifecycle', 'retirement_start_age', v)} />
                    <Slider label="다운사이징 확률" value={params.lifecycle.downsizing_probability * 100}
                      min={0} max={30} step={1} unit="%" onChange={v => update('lifecycle', 'downsizing_probability', v / 100)} />
                  </div>
                )}

                {advancedTab === 'network' && (
                  <div className="sliders">
                    <div className="section-title">Small-World 네트워크</div>
                    <Slider label="평균 이웃 수" value={params.network.avg_neighbors}
                      min={2} max={30} step={2} unit="명" onChange={v => update('network', 'avg_neighbors', v)} />
                    <Slider label="재연결 확률" value={params.network.rewiring_prob}
                      min={0} max={0.5} step={0.05} unit="" onChange={v => update('network', 'rewiring_prob', v)} />
                    <div className="section-title">정보 캐스케이드</div>
                    <Slider label="발동 임계값" value={params.network.cascade_threshold * 100}
                      min={10} max={70} step={5} unit="%" onChange={v => update('network', 'cascade_threshold', v / 100)} />
                    <Slider label="배율" value={params.network.cascade_multiplier}
                      min={1} max={5} step={0.5} unit="x" onChange={v => update('network', 'cascade_multiplier', v)} />
                    <Slider label="자기 신호 가중치" value={params.network.self_weight}
                      min={0.3} max={0.9} step={0.1} unit="" onChange={v => update('network', 'self_weight', v)} />
                  </div>
                )}

                {advancedTab === 'loan' && (
                  <div className="sliders">
                    <div className="section-title">LTV</div>
                    <Slider label="1주택자" value={params.policy.ltv_1house * 100} min={20} max={80} step={5} unit="%" onChange={v => update('policy', 'ltv_1house', v / 100)} />
                    <Slider label="2주택자" value={params.policy.ltv_2house * 100} min={0} max={60} step={5} unit="%" onChange={v => update('policy', 'ltv_2house', v / 100)} />
                    <Slider label="3주택+" value={params.policy.ltv_3house * 100} min={0} max={40} step={5} unit="%" onChange={v => update('policy', 'ltv_3house', v / 100)} />
                    <div className="section-title">DTI/DSR</div>
                    <Slider label="DTI 한도" value={params.policy.dti_limit * 100} min={20} max={70} step={5} unit="%" onChange={v => update('policy', 'dti_limit', v / 100)} />
                    <Slider label="DSR 한도" value={params.policy.dsr_limit * 100} min={20} max={70} step={5} unit="%" onChange={v => update('policy', 'dsr_limit', v / 100)} />
                    <div className="section-title">금리</div>
                    <Slider label="기준금리" value={params.policy.interest_rate * 100} min={1} max={8} step={0.25} unit="%" onChange={v => update('policy', 'interest_rate', v / 100)} />
                    <Slider label="모기지 스프레드" value={params.policy.mortgage_spread * 100} min={0.5} max={3} step={0.25} unit="%p" onChange={v => update('policy', 'mortgage_spread', v / 100)} />
                  </div>
                )}

                {advancedTab === 'tax' && (
                  <div className="sliders">
                    <div className="section-title">취득세</div>
                    <Slider label="1주택자" value={params.policy.acq_tax_1house * 100} min={0.5} max={5} step={0.5} unit="%" onChange={v => update('policy', 'acq_tax_1house', v / 100)} />
                    <Slider label="2주택자" value={params.policy.acq_tax_2house * 100} min={1} max={15} step={1} unit="%" onChange={v => update('policy', 'acq_tax_2house', v / 100)} />
                    <Slider label="3주택+" value={params.policy.acq_tax_3house * 100} min={1} max={20} step={1} unit="%" onChange={v => update('policy', 'acq_tax_3house', v / 100)} />
                    <div className="section-title">양도세</div>
                    <Slider label="단기 (2년-)" value={params.policy.transfer_tax_short * 100} min={30} max={80} step={5} unit="%" onChange={v => update('policy', 'transfer_tax_short', v / 100)} />
                    <Slider label="장기 (2년+)" value={params.policy.transfer_tax_long * 100} min={10} max={60} step={5} unit="%" onChange={v => update('policy', 'transfer_tax_long', v / 100)} />
                    <div className="section-title">종부세</div>
                    <Slider label="세율" value={params.policy.jongbu_rate * 100} min={0.5} max={5} step={0.5} unit="%" onChange={v => update('policy', 'jongbu_rate', v / 100)} />
                    <Slider label="1주택 기준" value={params.policy.jongbu_threshold_1house / 10000} min={6} max={20} step={1} unit="억" onChange={v => update('policy', 'jongbu_threshold_1house', v * 10000)} />
                  </div>
                )}

                {advancedTab === 'macro' && (
                  <div className="sliders">
                    <Slider label="M2 증가율 (연)" value={params.macro.m2_growth * 100} min={2} max={25} step={1} unit="%" onChange={v => update('macro', 'm2_growth', v / 100)} />
                    <Slider label="GDP 성장률 (연)" value={params.macro.gdp_growth_mean * 100} min={-3} max={10} step={0.5} unit="%" onChange={v => update('macro', 'gdp_growth_mean', v / 100)} />
                    <Slider label="GDP 변동성" value={params.macro.gdp_growth_volatility * 100} min={0} max={5} step={0.5} unit="%" onChange={v => update('macro', 'gdp_growth_volatility', v / 100)} />
                    <Slider label="인플레 목표" value={params.macro.inflation_target * 100} min={0} max={5} step={0.5} unit="%" onChange={v => update('macro', 'inflation_target', v / 100)} />
                    <Slider label="소득-GDP 탄력성" value={params.macro.income_gdp_beta} min={0.3} max={1.5} step={0.1} unit="" onChange={v => update('macro', 'income_gdp_beta', v)} />
                  </div>
                )}

                {advancedTab === 'supply' && (
                  <div className="sliders">
                    <div className="section-title">신규 공급</div>
                    <Slider label="기본 공급률" value={params.supply.base_supply_rate * 1000} min={0.1} max={5} step={0.1} unit="‰" onChange={v => update('supply', 'base_supply_rate', v / 1000)} />
                    <div className="section-title">공급 탄력성</div>
                    <Slider label="강남" value={params.supply.elasticity_gangnam} min={0.05} max={1} step={0.05} unit="" onChange={v => update('supply', 'elasticity_gangnam', v)} />
                    <Slider label="서울" value={params.supply.elasticity_seoul} min={0.1} max={1.5} step={0.1} unit="" onChange={v => update('supply', 'elasticity_seoul', v)} />
                    <Slider label="경기" value={params.supply.elasticity_gyeonggi} min={0.5} max={3} step={0.1} unit="" onChange={v => update('supply', 'elasticity_gyeonggi', v)} />
                    <Slider label="지방" value={params.supply.elasticity_local} min={1} max={5} step={0.5} unit="" onChange={v => update('supply', 'elasticity_local', v)} />
                    <div className="section-title">재건축</div>
                    <Slider label="기본 확률" value={params.supply.redevelopment_base_prob * 1000} min={0} max={5} step={0.5} unit="‰" onChange={v => update('supply', 'redevelopment_base_prob', v / 1000)} />
                    <Slider label="건설 기간" value={params.supply.construction_period} min={12} max={48} step={6} unit="월" onChange={v => update('supply', 'construction_period', v)} />
                    <div className="section-title">노후화</div>
                    <Slider label="노후화율" value={params.depreciation.depreciation_rate * 1000} min={0.5} max={5} step={0.5} unit="‰" onChange={v => update('depreciation', 'depreciation_rate', v / 1000)} />
                  </div>
                )}

                {advancedTab === 'market' && (
                  <div className="sliders">
                    <div className="section-title">가격 결정</div>
                    <Slider label="수요/공급 민감도" value={params.market.price_sensitivity * 10000} min={1} max={20} step={1} unit="‱" onChange={v => update('market', 'price_sensitivity', v / 10000)} />
                    <Slider label="기대 가중치" value={params.market.expectation_weight * 1000} min={1} max={30} step={1} unit="‰" onChange={v => update('market', 'expectation_weight', v / 1000)} />
                    <Slider label="기본 상승률 (월)" value={params.market.base_appreciation * 100} min={0} max={1} step={0.1} unit="%" onChange={v => update('market', 'base_appreciation', v / 100)} />
                    <div className="section-title">의사결정</div>
                    <Slider label="매수 임계값" value={params.market.buy_threshold} min={0.1} max={0.5} step={0.05} unit="" onChange={v => update('market', 'buy_threshold', v)} />
                    <Slider label="매도 임계값" value={params.market.sell_threshold} min={0.1} max={0.6} step={0.05} unit="" onChange={v => update('market', 'sell_threshold', v)} />
                    <div className="section-title">풍선효과</div>
                    <Slider label="전파 속도" value={params.market.spillover_rate * 1000} min={1} max={30} step={1} unit="‰" onChange={v => update('market', 'spillover_rate', v / 1000)} />
                  </div>
                )}
              </div>
            </div>
          )}
        </section>

        {/* 시작 버튼 */}
        <button className="btn-launch" onClick={() => onStart(params)}>
          시뮬레이션 시작
        </button>
      </div>
    </div>
  );
}
