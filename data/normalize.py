"""Normalizer for the OlmoEarth Pretrain dataset."""

import json
from enum import Enum
from pathlib import Path

import numpy as np

from data.constants import Modality


class Strategy(Enum):
    """The strategy to use for normalization."""
    PREDEFINED = "predefined"
    COMPUTED = "computed"


class Normalizer:
    """Normalize the data using broadcasting for efficiency."""

    _NORM_CONFIGS_DIR = Path(__file__).parent / "norm_configs"

    def __init__(
        self,
        strategy: Strategy,
        std_multiplier: float = 2,
    ) -> None:
        """Initialize the normalizer.

        Args:
            strategy: PREDEFINED (min-max) or COMPUTED (mean-std).
            std_multiplier: Only for COMPUTED. Std multiplier for range (~90% coverage).
        """
        self.strategy = strategy
        self.std_multiplier = std_multiplier
        self._config = self._load_config()
        
        # Precompute arrays for broadcasting (called once at init)
        self._precomputed_arrays = self._precompute_normalization_arrays()

    def _load_config(self) -> dict:
        """Load normalization config from JSON."""
        if self.strategy == Strategy.PREDEFINED:
            config_file = "predefined.json"
        else:
            config_file = "computed.json"
        
        with open(self._NORM_CONFIGS_DIR / config_file, 'r') as f:
            return json.load(f)

    def _precompute_normalization_arrays(self) -> dict:
        """Precompute min/max or mean/std arrays for all registered modalities."""
        arrays = {}
        
        # 放弃被动遍历不可控的 JSON，改为遍历系统中合法注册的 Modality
        for modality_spec in Modality.values():
            modality_name = modality_spec.name
            
            # 如果当前模态在 JSON 中没有归一化参数，则跳过（过滤脏数据）
            if modality_name not in self._config:
                continue
                
            modality_norm_values = self._config[modality_name]
            arr1_vals = []
            arr2_vals = []
            
            # 严格按照物理波段顺序 (band_order) 提取数据，彻底消除 sorted() 带来的通道错位灾难
            for band in modality_spec.band_order:
                if band not in modality_norm_values:
                    raise ValueError(f"配置缺失: 波段 {band} 未在 {modality_name} 的 JSON 中找到")
                    
                if self.strategy == Strategy.PREDEFINED:
                    arr1_vals.append(modality_norm_values[band]["min"])
                    arr2_vals.append(modality_norm_values[band]["max"])
                else:
                    arr1_vals.append(modality_norm_values[band]["mean"])
                    arr2_vals.append(modality_norm_values[band]["std"])
                    
            arrays[modality_name] = (np.array(arr1_vals), np.array(arr2_vals))
        
        return arrays

    def normalize(self, modality_spec, data: np.ndarray) -> np.ndarray:
        """Normalize data using broadcasting.

        Args:
            modality_spec: ModalitySpec object with .name attribute
            data: Input array [..., C] where C is number of bands

        Returns:
            Normalized array with same shape as input
        """
        modality_name = modality_spec.name
        
        if modality_name not in self._precomputed_arrays:
            raise ValueError(f"Modality {modality_name} not found in config")
        
        arr1, arr2 = self._precomputed_arrays[modality_name]
        
        if self.strategy == Strategy.PREDEFINED:
            # (data - min) / (max - min)
            # Reshape for broadcasting: (1,1,1,C)
            min_arr = arr1.reshape([1] * (data.ndim - 1) + [-1])
            max_arr = arr2.reshape([1] * (data.ndim - 1) + [-1])
            return (data - min_arr) / (max_arr - min_arr)
        else:
            # 截断拉伸：min = mean - 2 * std, max = mean + 2 * std, out = (data - min) / (max - min) -> [0, 1]
            mean_arr = arr1.reshape([1] * (data.ndim - 1) + [-1])
            std_arr = arr2.reshape([1] * (data.ndim - 1) + [-1])
            min_arr = mean_arr - self.std_multiplier * std_arr
            max_arr = mean_arr + self.std_multiplier * std_arr
        return (data - min_arr) / (max_arr - min_arr)
