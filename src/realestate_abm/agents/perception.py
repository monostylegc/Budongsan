"""인지 레이어 - 제한된 정보, 탐색비용, 미디어

에이전트는 실제 가격에 직접 접근 불가.
4가지 정보 소스: 자기 동네, 적극 탐색, 이웃 정보, 미디어.
"""

import numpy as np


class PerceptionEngine:
    """인지 엔진 - 에이전트의 '아는 것' 관리"""

    def __init__(self, cfg):
        """cfg: PerceptionConfig"""
        self.cfg = cfg

    def update(self, agents, world, actual_prices: np.ndarray, rng: np.random.Generator):
        """매 스텝 인지 업데이트

        Args:
            agents: AgentPopulation
            world: RegionSet
            actual_prices: (n_regions,) 실제 지역 가격
            rng: 난수 생성기
        """
        d = agents.data
        n = agents.n
        nr = agents.n_regions
        cfg = self.cfg

        # 1. 정보 노후화 (모든 정보 1개월씩 오래됨)
        d.price_info_age += 1

        # 2. 자기 동네 정보 (항상 알지만 노이즈 + 지연) - 벡터화
        noise = rng.normal(0, cfg.own_region_noise, n).astype(np.float32)
        agent_idx = np.arange(n)
        d.known_prices[agent_idx, d.region] = actual_prices[d.region] * (1.0 + noise)
        d.price_info_age[agent_idx, d.region] = cfg.own_region_delay

        # 3. 적극 탐색 (트리거된 에이전트만)
        triggered = d.is_triggered == 1
        n_triggered = np.sum(triggered)
        if n_triggered > 0:
            triggered_idx = np.where(triggered)[0]
            # 탐색 피로 누적
            d.search_fatigue[triggered_idx] += cfg.search_fatigue_rate
            d.search_fatigue = np.clip(d.search_fatigue, 0, 1)

            # 탐색 지역 수 (피로가 쌓이면 줄어듦)
            n_search = np.maximum(
                1,
                (cfg.search_regions_max * (1 - d.search_fatigue[triggered_idx])).astype(int)
            )

            for idx, agent_i in enumerate(triggered_idx):
                # 인접 지역 우선 탐색
                home = d.region[agent_i]
                adj_scores = world.adjacency[home].copy()
                adj_scores[home] = 0  # 자기 지역 제외 (이미 알고 있음)

                # 상위 n_search개 지역 선택
                top_regions = np.argsort(adj_scores)[-n_search[idx]:]

                for r in top_regions:
                    noise = rng.normal(0, cfg.own_region_noise * 1.5)
                    d.known_prices[agent_i, r] = actual_prices[r] * (1.0 + noise)
                    d.price_info_age[agent_i, r] = 0

        # 4. 이웃 정보 전파 (샘플링 기반 - 전체 순회 대신 일부만)
        # 매 스텝 에이전트의 10%만 이웃 정보 갱신 (성능 최적화)
        sample_size = min(n // 10, 5000)
        sampled = rng.choice(n, sample_size, replace=False)
        for i in sampled:
            nn = d.num_neighbors[i]
            if nn == 0:
                continue
            # 이웃 1명만 랜덤 선택
            nb_idx = rng.integers(0, nn)
            nb = d.neighbors[i, nb_idx]
            if nb < 0:
                continue
            # 이웃이 아는 가격 중 신선한 것 전파
            fresh = d.price_info_age[nb] < 3
            stale = d.price_info_age[i] > d.price_info_age[nb] + 1
            update_mask = fresh & stale
            if np.any(update_mask):
                noise = rng.normal(0, cfg.neighbor_noise_add, nr).astype(np.float32)
                d.known_prices[i, update_mask] = d.known_prices[nb, update_mask] * (1.0 + noise[update_mask])
                d.price_info_age[i, update_mask] = d.price_info_age[nb, update_mask] + 1

        # 5. 미디어 (전체 에이전트에 동일 신호, 개인 해석 다름)
        # 상위 3개 변동 지역만 뉴스화
        if len(actual_prices) > 0:
            price_change_signal = rng.normal(0, cfg.media_noise, size=nr)
            media_prices = actual_prices * (1.0 + price_change_signal)

            # 모든 에이전트에 미디어 정보 (오래된 정보만 업데이트)
            old_info = d.price_info_age > cfg.info_decay_months
            for r in range(nr):
                mask = old_info[:, r]
                d.known_prices[mask, r] = media_prices[r]
                d.price_info_age[mask, r] = 2  # 미디어 정보는 2개월 지연으로 처리

        # 6. 정보 품질 감쇠 (오래된 정보는 부정확해짐)
        decay_mask = d.price_info_age > cfg.info_decay_months
        # 감쇠된 정보에 추가 노이즈
        decay_noise = rng.normal(0, cfg.info_decay_rate, size=(n, nr))
        d.known_prices[decay_mask] *= (1.0 + decay_noise[decay_mask])
