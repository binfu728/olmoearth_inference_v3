"""Visualization functions for inference results."""

from pathlib import Path

import numpy as np
import torch


def get_rgb_channels(modality_name: str, input_np: np.ndarray) -> np.ndarray:
    """动态提取RGB通道，支持多种模态
    
    Args:
        modality_name: 模态名称 (如 'sentinel2', 'sentinel1' 等)
        input_np: 输入数组，形状为 [H, W, T, C] 或 [H, W, C]
    
    Returns:
        RGB通道数组，形状为 [H, W, 3]
    """
    # 获取维度信息
    if len(input_np.shape) == 4:
        H, W, T, C = input_np.shape
        temporal_slice = input_np[:, :, 0, :]  # 取第一个时间点
    elif len(input_np.shape) == 3:
        H, W, C = input_np.shape
        temporal_slice = input_np
    else:
        return np.zeros((4, 4, 3))
    
    # 如果通道数 < 3，用第一个通道复制三份
    if C < 3:
        return np.stack([temporal_slice[..., 0]] * 3, axis=-1)
    
    # 尝试从 Modality 获取通道顺序
    try:
        from data.constants import Modality
        bands = Modality.get(modality_name).band_order
        
        # 根据模态类型选择RGB通道
        if "sentinel2" in modality_name.lower() or "s2" in modality_name.lower():
            # Sentinel-2: B04(红), B03(绿), B02(蓝)
            rgb_names = ["B04", "B03", "B02"]
        elif "sentinel1" in modality_name.lower() or "s1" in modality_name.lower():
            # Sentinel-1 雷达: 没有真RGB，尝试VV/VH显示为灰度
            if len(bands) >= 2:
                # 用前两个通道显示
                rgb = np.stack([temporal_slice[..., 0], temporal_slice[..., 1], 
                               (temporal_slice[..., 0] + temporal_slice[..., 1]) / 2], axis=-1)
                return rgb
            else:
                return np.stack([temporal_slice[..., 0]] * 3, axis=-1)
        elif any(x in modality_name.lower() for x in ["landsat", "lt"]):
            # Landsat: B4, B3, B2 或 B5, B4, B3
            rgb_names = ["B4", "B3", "B2"] if "B4" in bands else ["B5", "B4", "B3"]
        elif "modis" in modality_name.lower():
            # MODIS: B1, B4, B3
            rgb_names = ["B1", "B4", "B3"]
        else:
            # 默认取前三个通道
            return temporal_slice[..., :3]
        
        # 查找RGB通道索引
        idx = []
        for name in rgb_names:
            if name in bands:
                idx.append(bands.index(name))
            else:
                # 通道名不存在，回退到前三个通道
                return temporal_slice[..., :3]
        
        return temporal_slice[..., idx]
        
    except (ValueError, ImportError, AttributeError):
        # 回退方案：直接取前三个通道
        return temporal_slice[..., :3] if C >= 3 else np.stack([temporal_slice[..., 0]] * 3, axis=-1)


def visualize_multimodal_features(
    features_dict: dict,
    input_tensors_dict: dict,
    output_dir: str = "."
):
    """可视化多模态特征
    
    Args:
        features_dict: 模态名称到特征的字典 {modality: tensor}
        input_tensors_dict: 模态名称到输入张量的字典 {modality: tensor}
        output_dir: 输出目录
    """
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示: 需要安装 sklearn 和 matplotlib 来保存可视化结果")
        print("  pip install scikit-learn matplotlib")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_modalities = len(features_dict)
    fig, axes = plt.subplots(num_modalities, 2, figsize=(12, 6 * num_modalities))
    
    if num_modalities == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (modality, features) in enumerate(features_dict.items()):
        input_tensor = input_tensors_dict.get(modality)
        
        # 处理特征
        if features is None or len(features.shape) == 0:
            continue
        
        # 获取特征形状
        if len(features.shape) == 4:  # [B, H', W', D] pooled
            features_2d = features[0].cpu().numpy().reshape(-1, features.shape[-1])
        elif len(features.shape) == 6:  # [B, H', W', T, S, D]
            B, H_prime, W_prime, T, S, D = features.shape
            features_2d = features[0].detach().cpu().numpy().reshape(-1, D)
        else:
            continue
        
        # PCA降维
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_2d)
        
        # 重塑为图像
        if len(features.shape) == 6:
            features_img = features_pca.reshape(H_prime, W_prime, T, S, 3)
            features_img = features_img[:, :, 0, 0, :]
        else:
            H_prime, W_prime = features.shape[1], features.shape[2]
            features_img = features_pca.reshape(H_prime, W_prime, 3)
        
        # 归一化
        features_img = (features_img - features_img.min()) / (features_img.max() - features_img.min() + 1e-8)
        
        # 绘制特征图
        axes[idx, 0].imshow(features_img)
        axes[idx, 0].set_title(f"{modality} - PCA Features")
        axes[idx, 0].axis('off')
        
        # 绘制输入 (使用动态RGB通道提取)
        if input_tensor is not None:
            input_np = input_tensor[0].cpu().numpy()
            input_rgb = get_rgb_channels(modality, input_np)
            
            input_rgb = (input_rgb - input_rgb.min()) / (input_rgb.max() - input_rgb.min() + 1e-8)
            axes[idx, 1].imshow(input_rgb)
            axes[idx, 1].set_title(f"{modality} - Input RGB")
            axes[idx, 1].axis('off')
        else:
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "multimodal_feature_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"多模态可视化结果已保存: {output_path}")


def save_features_dict_to_file(features_dict: dict, output_dir: str | Path):
    """保存多模态特征字典到文件
    
    Args:
        features_dict: 特征字典
        output_dir: 输出目录
    """
    import pickle
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_np_dict = {}
    for modality, features in features_dict.items():
        if features is not None:
            features_np_dict[modality] = features.detach().cpu().numpy()
    
    output_path = output_dir / "features.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(features_np_dict, f)
    
    print(f"多模态特征已保存到: {output_path}")