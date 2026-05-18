import argparse
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from dataload.h5_loader import (
    MultiModalEarthDataset,
    multimodal_collate_fn,
)
from dataload.model import load_config, load_model_direct, load_model_with_weights
from dataload.visualization import visualize_multimodal_features, save_features_dict_to_file


def run_inference(
    config: dict,
    metadata_path: str | Path,
    h5_dir: str | Path,
    device: str = "cuda",
    patch_size: int = 4,
    batch_size: int = 1,
    output_dir: str = ".",
    max_samples: int = None,
    save_features: bool = True,
    visualize: bool = True,
    weights_path: str | Path = None,
) -> List[Dict]:
    """使用DataLoader批量推理"""
    metadata_path = Path(metadata_path)
    h5_dir = Path(h5_dir)
    
    print("=" * 60)
    print("OlmoEarth 多模态推理")
    print("=" * 60)
    
    dataset = MultiModalEarthDataset(metadata_path, h5_dir, config, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=multimodal_collate_fn, num_workers=0)
    
    print("\n加载模型...")
    model = load_model_direct(config)
    model = model.to(device)
    model.eval()
    
    if weights_path is None:
        weights_path = Path(__file__).parent / "params" / "weights.pth"
    model = load_model_with_weights(model, weights_path)
    
    results_list = []
    device_obj = torch.device(device)
    
    print(f"\n开始推理 (batch_size={batch_size})...")
    
    with torch.no_grad():
        for i, batch_sample in enumerate(dataloader):
            if max_samples is not None and i >= max_samples:
                break
            
            print(f"\n处理样本 {i}...")
            sample = {k: v.to(device_obj) for k, v in batch_sample.items()}
            
            output_dict = model(sample, patch_size=patch_size)
            
            # 1. 提取特征并对齐老版本逻辑
            extracted_features = {}
            features_for_vis = {}
            
            modalities = [k for k in output_dict.keys() if not k.endswith("_mask")]
            
            for mod_name in modalities:
                mod_features = output_dict[mod_name]
                
                pooled = mod_features.mean(dim=[3, 4])
                extracted_features[mod_name] = pooled
                features_for_vis[mod_name] = mod_features
                
                print(f"  {mod_name}: pooled={pooled.shape}")
            
            # 2. 融合
            if extracted_features:
                all_features = list(extracted_features.values())
                if len(all_features) > 1:
                    fused = torch.stack(all_features).mean(dim=0)
                    extracted_features['fused'] = fused
                    print(f"  fused: shape={fused.shape}")
            
            # 3. 保存/可视化
            if save_features or visualize:
                input_dict = {k: v.cpu() for k, v in sample.items() if not k.endswith("_mask") and k != "timestamps"}
                
                if save_features:
                    features_cpu = {k: v.cpu() for k, v in features_for_vis.items()}
                    save_features_dict_to_file(features_cpu, output_dir)
                
                if visualize:
                    visualize_multimodal_features(features_for_vis, input_dict, output_dir)
            
            results_list.append(extracted_features)
    
    print("\n" + "=" * 60)
    print(f"推理完成，共处理 {len(results_list)} 个样本")
    print("=" * 60)
    return results_list


def main():
    parser = argparse.ArgumentParser(description="OlmoEarth 多模态推理")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认使用 params/config.json）")
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else Path(__file__).parent / "params" / "config.json"
    config = load_config(config_path)
    
    inference_cfg = config.get('inference', {})
    paths_cfg = config.get('paths', {})
    
    metadata = inference_cfg.get('metadata')
    h5_dir = inference_cfg.get('h5_dir')
    if metadata is None or h5_dir is None:
        raise ValueError("inference.metadata 或 h5_dir 未设置，请修改 config.json")
    
    run_inference(
        config=config,
        metadata_path=metadata,
        h5_dir=h5_dir,
        device=inference_cfg.get('device', 'cuda'),
        patch_size=inference_cfg.get('patch_size', 4),
        batch_size=inference_cfg.get('batch_size', 1),
        output_dir=inference_cfg.get('output_dir', 'results'),
        max_samples=inference_cfg.get('max_samples'),
        save_features=inference_cfg.get('save_features', True),
        visualize=inference_cfg.get('visualize', True),
        weights_path=paths_cfg.get('weights'),
    )


if __name__ == "__main__":
    main()