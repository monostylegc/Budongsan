"""거리 및 인접도 계산"""

import numpy as np


def haversine_distance(coords: np.ndarray) -> np.ndarray:
    """위경도 좌표에서 거리 행렬 계산 (km)

    Args:
        coords: (n, 2) 배열 [경도, 위도]

    Returns:
        (n, n) 거리 행렬 (km)
    """
    n = len(coords)
    R = 6371.0  # 지구 반경 (km)

    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])

    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]

    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return (R * c).astype(np.float32)


def compute_distances(coords: np.ndarray) -> np.ndarray:
    """좌표에서 거리 행렬 계산

    좌표값이 100 이상이면 위경도로 간주하여 haversine 사용,
    아니면 유클리드 거리 사용
    """
    if np.any(np.abs(coords) > 100):
        return haversine_distance(coords)
    else:
        # 유클리드 거리 (km 단위 가정)
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)


def compute_adjacency(distances: np.ndarray, decay: float = 0.03) -> np.ndarray:
    """거리에서 인접도 계산

    adjacency[i][j] = exp(-decay * distance[i][j])

    Args:
        distances: (n, n) 거리 행렬
        decay: 감쇠율 (클수록 빠르게 감소)

    Returns:
        (n, n) 인접도 행렬 (0~1)
    """
    adj = np.exp(-decay * distances)
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)
