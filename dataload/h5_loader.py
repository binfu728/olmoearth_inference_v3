"""H5 data loading for OlmoEarth inference."""

import pandas as pd
from pathlib import Path
from typing import Dict

import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset

from data.constants import MISSING_VALUE, Modality, MaskValue
from data.normalize import Normalizer, Strategy


class MultiModalEarthDataset(Dataset):
    """Load H5 data - dynamic, pure satellite optimized."""
    
    def __init__(
        self, 
        metadata_path: str | Path, 
        h5_dir: str | Path,
        config_dict: dict,
        patch_size: int = 4
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.h5_dir = Path(h5_dir)
        self.patch_size = patch_size
        
        self.modalities = config_dict.get('modalities', {}).get('inference_use', [])
        self.default_T = config_dict.get('model', {}).get('max_timesteps', 12)
        norm_strat_str = config_dict.get('inference', {}).get('normalize_strategy', 'computed')
        
        strategy = Strategy.PREDEFINED if norm_strat_str == "predefined" else Strategy.COMPUTED
        self.normalizer = Normalizer(strategy=strategy)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        sample_idx = row['sample_index']
        h5_path = self.h5_dir / f"sample_{sample_idx}.h5"
        
        sample_kwargs = {}
        
        if not h5_path.exists():
            raise FileNotFoundError(f"H5文件未找到: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            # 【修复】动态获取 H, W，避免硬编码的 IMAGE_TILE_SIZE=256 导致维度不匹配
            detected_H, detected_W = None, None
            
            for mod in self.modalities:
                if mod in row and row[mod] == 1 and mod in f:
                    data_shape = f[mod].shape  # [H, W, T, C] 或 [H, W, C]
                    detected_H, detected_W = data_shape[0], data_shape[1]
                    break
            
            for mod in self.modalities:
                mod_spec = Modality.get(mod)
                num_bandsets, num_bands = mod_spec.num_band_sets, mod_spec.num_bands
                
                if mod in row and row[mod] == 1 and mod in f:
                    data = f[mod][:]
                    data = np.where(data == MISSING_VALUE, 0.0, data)
                    
                    try:
                        data = self.normalizer.normalize(mod_spec, data[np.newaxis, ...].astype(np.float32))[0]
                    except Exception:
                        pass
                        
                    tensor = torch.tensor(data, dtype=torch.float32)
                    H, W, T, C = tensor.shape
                    mask = torch.full((H, W, T, num_bandsets), MaskValue.ONLINE_ENCODER.value, dtype=torch.long)
                    
                    sample_kwargs[mod] = tensor
                    sample_kwargs[f"{mod}_mask"] = mask
                else:
                    # 【修复】使用动态检测的 H, W，而非硬编码的 get_expected_tile_size()
                    H = detected_H if detected_H is not None else 256
                    W = detected_W if detected_W is not None else 256
                    sample_kwargs[mod] = torch.zeros((H, W, self.default_T, num_bands), dtype=torch.float32)
                    sample_kwargs[f"{mod}_mask"] = torch.full((H, W, self.default_T, num_bandsets), MaskValue.MISSING.value, dtype=torch.long)
            
            if 'timestamps' not in f:
                raise KeyError("H5文件中缺少'timestamps'字段")
            ts = torch.tensor(f['timestamps'][:], dtype=torch.int64) # 月份已经为0-indexed
            sample_kwargs['timestamps'] = ts
        
        self._apply_time_padding(sample_kwargs)
        return sample_kwargs
    
    def _apply_time_padding(self, sample_kwargs: Dict):
        """统一暴力对齐 T 维度"""
        ts = sample_kwargs.get('timestamps')
        if ts is None: return
        target_T = self.default_T

        for mod in self.modalities:
            if mod in sample_kwargs:
                data = sample_kwargs[mod]
                mask = sample_kwargs.get(f"{mod}_mask")
                current_T = data.shape[2]
                
                if current_T >= target_T:
                    sample_kwargs[mod] = data[:, :, :target_T, :]
                    if mask is not None:
                        sample_kwargs[f"{mod}_mask"] = mask[:, :, :target_T, :]
                else:
                    pad_T = target_T - current_T
                    pad_data = torch.zeros(data.shape[0], data.shape[1], pad_T, data.shape[3], dtype=data.dtype)
                    sample_kwargs[mod] = torch.cat([data, pad_data], dim=2)
                    if mask is not None:
                        pad_mask = torch.full((mask.shape[0], mask.shape[1], pad_T, mask.shape[3]), MaskValue.MISSING.value, dtype=mask.dtype)
                        sample_kwargs[f"{mod}_mask"] = torch.cat([mask, pad_mask], dim=2)

        ts_T = ts.shape[0]
        if ts_T >= target_T:
            sample_kwargs['timestamps'] = ts[:target_T, :]
        else:
            pad_ts = torch.zeros(target_T - ts_T, ts.shape[1], dtype=ts.dtype)
            sample_kwargs['timestamps'] = torch.cat([ts, pad_ts], dim=0)


def multimodal_collate_fn(batch_list):
    """原生 Dict Collate: 返回纯净的 Dict[str, Tensor]"""
    batched = {}
    for sample in batch_list:
        for key, value in sample.items():
            batched.setdefault(key, []).append(value)
    return {k: torch.stack(v) for k, v in batched.items()}