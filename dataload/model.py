"""Model loading and utility functions."""

import json
from pathlib import Path
import torch
from data.constants import Modality

def load_config(config_path: str | Path = None) -> dict:
    """从配置文件加载配置"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "params" / "config.json"
    
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def load_model_direct(config_dict: dict):
    """直接从传入的配置字典构建模型 (极致精简版)"""
    from nn.flexi_vit import Encoder
    from data.constants import get_modality_specs_from_names
    
    # 1. 提取基础网络参数
    model_config = config_dict.get('model', {})
    
    # 2. 获取支持的模态字符串列表
    supported_names = config_dict.get('modalities', {}).get('supported', Modality.names())
    
    # 3. 转换并注入 supported_modalities 对象列表
    model_config['supported_modalities'] = get_modality_specs_from_names(supported_names)
    
    # 4. 字典解包，直接构建模型！
    encoder = Encoder(**model_config)
    print("模型构建成功（仅 Encoder）")
    
    return encoder

def load_model_with_weights(model, weights_path: str | Path = None):
    """加载模型权重，支持动态键名映射"""
    if weights_path is None:
        weights_path = Path(__file__).parent.parent / "params" / "weights.pth"
    
    weights_path = Path(weights_path)
    if not weights_path.exists() or weights_path.stat().st_size <= 1000:
        print("提示: 未找到权重文件，模型使用随机初始化权重")
        return model
    
    print(f"加载权重文件: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 解包 PyTorch Lightning 格式
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # 去除可能的模型编译前缀
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # 建立新的映射字典
    new_state_dict = {}
    skipped_keys = []
    
    for k, v in state_dict.items():
        # 丢弃训练相关模块
        if k.startswith('decoder.') or k.startswith('target_encoder.'):
            skipped_keys.append(k)
            continue
        
        # 丢弃 project_and_aggregate 训练用投影层
        if k.startswith('project_and_aggregate'):
            skipped_keys.append(k)
            continue
        
        # 去除 encoder. 前缀
        new_key = k.replace('encoder.', '') if k.startswith('encoder.') else k
        
        # 去除其他分布式训练前缀
        new_key = new_key.replace('module.', '').replace('_forward_module.', '').replace('model.module.', '')
        
        # 处理 embedding_projector 键名映射
        if 'embedding_projector.projection.0.' in new_key:
            new_key = new_key.replace('.projection.0.', '.')
        
        new_state_dict[new_key] = v
    
    print(f"跳过 {len(skipped_keys)} 个训练相关权重 (decoder/target_encoder/project_and_aggregate)")
    
    # 加载权重
    model.load_state_dict(new_state_dict, strict=False)
    print("权重加载完成")
    
    return model