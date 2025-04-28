import os
import torch
import clip as clip
import argparse
from tqdm import tqdm

def download_clip(model_name, save_dir=None):
    """
    下载CLIP模型到指定目录
    
    Args:
        model_name: CLIP模型名称，如 'ViT-B/32'
        save_dir: 保存目录，默认为当前脚本所在目录
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"正在下载 {model_name} 模型...")
    
    try:
        # 加载模型，这会自动下载
        model, preprocess = clip.load(model_name, device="cpu", download_root=save_dir)
        
        # 保存模型
        model_path = os.path.join(save_dir, f"{model_name.replace('/', '-')}.pt")
        if not os.path.exists(model_path):
            torch.save(model.state_dict(), model_path)
        
        print(f"CLIP模型已下载并保存到 {model_path}")
        return model_path
    except Exception as e:
        print(f"下载CLIP模型时出错: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='下载CLIP模型')
    parser.add_argument('--model_name', type=str, default='ViT-B/32', 
                        help='CLIP模型名称, 支持 ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"]')
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='保存目录，默认为当前脚本所在目录')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    download_clip(args.model_name, args.save_dir)