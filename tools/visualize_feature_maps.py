import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径，使脚本可以从任何位置运行
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import zoom

from models import helper
from vpr_model import VPRModel


def load_image(image_path: str, image_size=(224, 224)) -> torch.Tensor:
    """加载并预处理图像"""
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """将torch tensor转换为numpy数组"""
    return t.detach().cpu().numpy()


def normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max归一化到[0, 1]"""
    min_v = x.min()
    max_v = x.max()
    return (x - min_v) / (max_v - min_v + eps)


def overlay_heatmap_on_image(base_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.3, cmap: str = 'jet',
                             gamma: float = 1.0) -> np.ndarray:
    """
    将热力图叠加在原图上

    Args:
        base_img: [H, W, 3] 原图，值域[0,1]
        heatmap: [H, W] 热力图，值域[0,1]
        alpha: 叠加透明度（0-1，越小热图越淡）
        cmap: 颜色映射
        gamma: gamma校正值，用于调整热图强度（<1变亮，>1变暗）
    """
    # 应用gamma校正来调整热图强度
    if gamma != 1.0:
        heatmap = np.power(heatmap, gamma)

    cm = plt.get_cmap(cmap)
    colored = cm(heatmap)[..., :3]  # drop alpha
    out = (1 - alpha) * base_img + alpha * colored
    return np.clip(out, 0, 1)


def select_channels_by_variance(feature_map: torch.Tensor, k: int) -> np.ndarray:
    """
    根据空间方差选择前k个特征通道

    Args:
        feature_map: [1, C, H, W] 特征图
        k: 要选择的通道数
    """
    C = feature_map.shape[1]
    k = min(k, C)
    f_np = to_numpy(feature_map[0])  # [C, H, W]
    variances = f_np.reshape(C, -1).var(axis=1)
    idx = np.argsort(-variances)[:k]
    return idx


def load_ckpt_flex(model, ckpt_path,
                   strip_prefixes=('state_dict.', 'model.', 'module.'),
                   replace_prefixes=(('backbone.model.', 'backbone.'),
                                     ('aggregator.', 'aggregator.')),
                   verbose=True):
    """
    让 Lightning/分布式/命名差异的 ckpt 也能稳妥加载：
      1) 取 ckpt['state_dict']（若无则用原字典）
      2) 去掉多余前缀（state_dict./model./module.）
      3) 前缀替换（如 backbone.model. -> backbone.）
      4) 过滤掉形状不符/模型里没有的键
      5) strict=False 加载
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

    new_sd = {}
    for k, v in sd.items():
        nk = k
        for p in strip_prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        for old, new in replace_prefixes:
            if nk.startswith(old):
                nk = new + nk[len(old):]
        new_sd[nk] = v

    model_sd = model.state_dict()
    filtered = {}
    skipped_shape = []
    for k, v in new_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in model_sd:
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    missing = [k for k in model_sd.keys() if k not in filtered]
    unexpected = [k for k in new_sd.keys() if k not in model_sd]

    if verbose:
        print(
            f"[load_ckpt_flex] loadable={len(filtered)} | missing={len(missing)} | unexpected={len(unexpected)} | shape_mismatch={len(skipped_shape)}")
        if skipped_shape[:5]:
            print("  shape-mismatch(ex):", skipped_shape[:5])
        if missing[:5]:
            print("  missing(ex):", missing[:5])
        if unexpected[:5]:
            print("  unexpected(ex):", unexpected[:5])

    msg = model.load_state_dict(filtered, strict=False)
    if verbose:
        print("[load_ckpt_flex] load_state_dict msg:", msg)
    return msg


def load_pretrained_model(ckpt_path: str, device: str,
                          backbone_arch: str = 'dinov2_vitb14',
                          backbone_config: dict | None = None,
                          agg_arch: str = 'SALAD',
                          agg_config: dict | None = None):
    """
    从checkpoint加载预训练的VPR模型，并返回backbone用于特征提取

    Args:
        ckpt_path: checkpoint文件路径
        device: 设备 ('cuda' 或 'cpu')
        backbone_arch: backbone架构名称
        backbone_config: backbone配置字典
        agg_arch: aggregator架构名称
        agg_config: aggregator配置字典

    Returns:
        backbone: 加载了预训练权重的backbone模型
    """
    if backbone_config is None:
        backbone_config = {
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        }

    if agg_config is None:
        # 根据backbone自动推断num_channels
        if 'dinov2_vitb' in backbone_arch.lower():
            num_channels = 768
        elif 'dinov2_vitl' in backbone_arch.lower():
            num_channels = 1024
        elif 'dinov2_vitg' in backbone_arch.lower():
            num_channels = 1536
        elif 'dinov2_vits' in backbone_arch.lower():
            num_channels = 384
        else:
            num_channels = 768  # 默认值

        agg_config = {
            'num_channels': num_channels,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        }

    # 构建完整的VPR模型
    model = VPRModel(
        backbone_arch=backbone_arch,
        backbone_config=backbone_config,
        agg_arch=agg_arch,
        agg_config=agg_config,
    ).to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 加载权重
    load_ckpt_flex(
        model,
        ckpt_path,
        strip_prefixes=('state_dict.', 'model.', 'module.'),
        replace_prefixes=(('backbone.model.', 'backbone.'), ('aggregator.', 'aggregator.')),
        verbose=True
    )

    model.eval()
    print(f"Loaded pretrained model from {ckpt_path} successfully!")

    # 返回backbone用于特征提取
    return model.backbone


def visualize_feature_maps(
        image_path: str,
        output_dir: str,
        backbone_arch: str = 'dinov2_vitb14',
        backbone_config: dict | None = None,
        num_channels_grid: int = 16,
        image_size=(224, 224),
        colormap: str = 'jet',
        ckpt_path: str | None = None,
        agg_arch: str = 'SALAD',
        agg_config: dict | None = None,
        heatmap_only: bool = False,
        alpha: float = 0.3,
        gamma: float = 1.0,
        output_dpi: int = 300,
        upscale_factor: int = 4
):
    """
    可视化特征响应图

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        backbone_arch: backbone架构名称
        backbone_config: backbone配置字典
        num_channels_grid: 网格中显示的特征通道数
        image_size: 输入图像尺寸
        colormap: 颜色映射
        ckpt_path: 预训练checkpoint路径（如果提供，将从checkpoint加载权重）
        agg_arch: aggregator架构（仅在ckpt_path提供时使用）
        agg_config: aggregator配置（仅在ckpt_path提供时使用）
        heatmap_only: 是否只生成heatmap可视化
        alpha: 叠加透明度（0-1，越小热图越淡）
        gamma: gamma校正值（<1变亮，>1变暗）
    """
    # 确保输出目录是绝对路径
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Output directory: {output_dir}")

    if backbone_config is None:
        backbone_config = {}

    # 如果提供了checkpoint路径，从预训练模型加载backbone
    if ckpt_path is not None:
        print(f"Loading pretrained weights from: {ckpt_path}")
        backbone = load_pretrained_model(
            ckpt_path=ckpt_path,
            device=device,
            backbone_arch=backbone_arch,
            backbone_config=backbone_config,
            agg_arch=agg_arch,
            agg_config=agg_config
        )
    else:
        # 否则使用默认的backbone（无预训练权重或使用backbone自带的预训练权重）
        print("Using backbone without loading checkpoint (may use ImageNet pretrained weights)")
        backbone = helper.get_backbone(backbone_arch, backbone_config).to(device).eval()

    image = load_image(image_path, image_size=image_size).to(device)

    with torch.no_grad():
        out = backbone(image)

    # Some backbones (DINOv2 with return_token=True) may return (features, token)
    if isinstance(out, (tuple, list)):
        features = out[0]
    else:
        features = out

    # features: [1, C, Hf, Wf]
    features_np = to_numpy(features[0])  # [C, Hf, Wf]
    C, Hf, Wf = features_np.shape

    print(f"Feature map shape: {C} channels, {Hf}x{Wf} spatial size")

    # 计算放大后的尺寸（用于高质量输出）
    Hf_upscaled = Hf * upscale_factor
    Wf_upscaled = Wf * upscale_factor

    # Base image for overlay (resize to upscaled feature spatial size for better quality)
    base_img = Image.open(image_path).convert('RGB')
    base_img_orig = base_img.copy()
    base_img_small = base_img.resize((Wf, Hf))
    base_img_large = base_img.resize((Wf_upscaled, Hf_upscaled), Image.Resampling.LANCZOS)
    base_np = np.asarray(base_img_small).astype(np.float32) / 255.0
    base_np_large = np.asarray(base_img_large).astype(np.float32) / 255.0

    # 1) Averaged heatmap overlay
    avg_map = features_np.mean(axis=0)
    avg_map = normalize_minmax(avg_map)
    # 放大特征图到更高分辨率
    avg_map_large = zoom(avg_map, (upscale_factor, upscale_factor), order=1)
    overlay = overlay_heatmap_on_image(base_np, avg_map, alpha=alpha, cmap=colormap, gamma=gamma)
    overlay_large = overlay_heatmap_on_image(base_np_large, avg_map_large, alpha=alpha, cmap=colormap, gamma=gamma)
    # 使用高DPI保存
    fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
    ax.imshow(overlay_large)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(output_dir, 'feature_avg_overlay.png'), dpi=output_dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 1.1) Save averaged heatmap with colorbar (standalone heatmap)
    fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
    im = ax.imshow(avg_map_large, cmap=colormap, aspect='auto')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Activation Intensity')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'heatmap_avg_with_colorbar.png'), dpi=output_dpi, bbox_inches='tight')
    plt.close(fig)

    # 1.2) Save averaged heatmap (pure heatmap, no colorbar)
    fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
    ax.imshow(avg_map_large, cmap=colormap, aspect='auto')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(output_dir, 'heatmap_avg.png'), dpi=output_dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 1.3) Max activation heatmap
    max_map = features_np.max(axis=0)
    max_map = normalize_minmax(max_map)
    max_map_large = zoom(max_map, (upscale_factor, upscale_factor), order=1)
    fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
    ax.imshow(max_map_large, cmap=colormap, aspect='auto')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(output_dir, 'heatmap_max.png'), dpi=output_dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    overlay_max = overlay_heatmap_on_image(base_np_large, max_map_large, alpha=alpha, cmap=colormap, gamma=gamma)
    fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
    ax.imshow(overlay_max)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(output_dir, 'heatmap_max_overlay.png'), dpi=output_dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 1.4) L2 norm heatmap
    l2_map = np.sqrt((features_np ** 2).sum(axis=0))
    l2_map = normalize_minmax(l2_map)
    l2_map_large = zoom(l2_map, (upscale_factor, upscale_factor), order=1)
    fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
    ax.imshow(l2_map_large, cmap=colormap, aspect='auto')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(output_dir, 'heatmap_l2.png'), dpi=output_dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    overlay_l2 = overlay_heatmap_on_image(base_np_large, l2_map_large, alpha=alpha, cmap=colormap, gamma=gamma)
    fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
    ax.imshow(overlay_l2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(output_dir, 'heatmap_l2_overlay.png'), dpi=output_dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 2) Grid of selected channels (by spatial variance) - skip if heatmap_only
    if not heatmap_only:
        sel_idx = select_channels_by_variance(features, num_channels_grid)
        grid_cols = int(np.ceil(np.sqrt(len(sel_idx))))
        grid_rows = int(np.ceil(len(sel_idx) / grid_cols))
        fig_h = max(2, grid_rows)
        fig_w = max(2, grid_cols)
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3 * fig_w, 3 * fig_h))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(grid_rows, grid_cols)

        for i in range(grid_rows * grid_cols):
            ax = axes[i // grid_cols, i % grid_cols]
            ax.axis('off')
            if i < len(sel_idx):
                ch = sel_idx[i]
                ch_map = normalize_minmax(features_np[ch])
                ax.imshow(ch_map, cmap=colormap)
                ax.set_title(f'ch {ch}', fontsize=8)
            else:
                ax.imshow(np.zeros((Hf, Wf)))

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'feature_grid.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

        # 3) Save raw averaged heatmap as grayscale
        fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
        ax.imshow(avg_map_large, cmap='gray', aspect='auto')
        ax.axis('off')
        plt.tight_layout(pad=0)
        fig.savefig(os.path.join(output_dir, 'feature_avg_gray.png'), dpi=output_dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # 4) Save individual top channels with heatmaps
        top_n = min(5, len(sel_idx))
        for i, ch_idx in enumerate(sel_idx[:top_n]):
            ch_map = normalize_minmax(features_np[ch_idx])
            ch_map_large = zoom(ch_map, (upscale_factor, upscale_factor), order=1)
            # Overlay version
            overlay_ch = overlay_heatmap_on_image(base_np_large, ch_map_large, alpha=alpha, cmap=colormap, gamma=gamma)
            fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
            ax.imshow(overlay_ch)
            ax.axis('off')
            plt.tight_layout(pad=0)
            fig.savefig(os.path.join(output_dir, f'feature_ch{ch_idx}_overlay.png'), dpi=output_dpi,
                        bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            # Pure heatmap version
            fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
            ax.imshow(ch_map_large, cmap=colormap, aspect='auto')
            ax.axis('off')
            plt.tight_layout(pad=0)
            fig.savefig(os.path.join(output_dir, f'heatmap_ch{ch_idx}.png'), dpi=output_dpi, bbox_inches='tight',
                        pad_inches=0)
            plt.close(fig)
            # Heatmap with colorbar
            fig, ax = plt.subplots(figsize=(Wf_upscaled / 100, Hf_upscaled / 100), dpi=100)
            im = ax.imshow(ch_map_large, cmap=colormap, aspect='auto')
            ax.axis('off')
            ax.set_title(f'Channel {ch_idx} Heatmap', fontsize=12, pad=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Activation')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'heatmap_ch{ch_idx}_with_colorbar.png'), dpi=output_dpi,
                        bbox_inches='tight')
            plt.close(fig)
    else:
        # For heatmap_only mode, still need sel_idx for comparison figure
        sel_idx = select_channels_by_variance(features, min(5, num_channels_grid))

    # 5) Create a comprehensive heatmap comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(Wf_upscaled / 50, Hf_upscaled / 50), dpi=100)
    axes = axes.flatten()

    # Average
    im0 = axes[0].imshow(avg_map_large, cmap=colormap, aspect='auto')
    axes[0].set_title('Average Activation', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Max
    im1 = axes[1].imshow(max_map_large, cmap=colormap, aspect='auto')
    axes[1].set_title('Max Activation', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # L2 norm
    im2 = axes[2].imshow(l2_map_large, cmap=colormap, aspect='auto')
    axes[2].set_title('L2 Norm', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Top channel
    if len(sel_idx) > 0:
        top_ch_map = normalize_minmax(features_np[sel_idx[0]])
        top_ch_map_large = zoom(top_ch_map, (upscale_factor, upscale_factor), order=1)
        im3 = axes[3].imshow(top_ch_map_large, cmap=colormap, aspect='auto')
        axes[3].set_title(f'Top Channel {sel_idx[0]}', fontsize=12)
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'heatmap_comparison.png'), dpi=output_dpi, bbox_inches='tight')
    plt.close(fig)

    print(f'\n✓ Saved visualizations to: {output_dir}')
    print(f'\n📊 Heatmap Files:')
    print(f'  - heatmap_avg.png (平均heatmap)')
    print(f'  - heatmap_avg_with_colorbar.png (平均heatmap带colorbar)')
    print(f'  - heatmap_max.png (最大值heatmap)')
    print(f'  - heatmap_max_overlay.png (最大值heatmap叠加)')
    print(f'  - heatmap_l2.png (L2范数heatmap)')
    print(f'  - heatmap_l2_overlay.png (L2范数heatmap叠加)')
    print(f'  - heatmap_comparison.png (多种heatmap对比)')
    print(f'  - feature_avg_overlay.png (平均特征响应图叠加)')
    if not heatmap_only:
        print(f'\n📈 Additional Visualizations:')
        print(f'  - feature_avg_gray.png (平均特征响应图灰度)')
        print(f'  - feature_grid.png (前{num_channels_grid}个通道网格)')
        if len(sel_idx) > 0:
            top_n = min(5, len(sel_idx))
            print(f'  - feature_ch*_overlay.png (前{top_n}个通道单独叠加图)')
            print(f'  - heatmap_ch*.png (前{top_n}个通道纯heatmap)')
            print(f'  - heatmap_ch*_with_colorbar.png (前{top_n}个通道heatmap带colorbar)')


def parse_backbone_config_from_kv(args_kv: list[str]) -> dict:
    """从key=value格式的字符串列表解析配置字典"""
    config: dict = {}
    for kv in args_kv:
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        k = k.strip()
        v = v.strip()
        # naive parsing for ints/bools
        if v.lower() in ['true', 'false']:
            config[k] = v.lower() == 'true'
        else:
            try:
                config[k] = int(v)
            except ValueError:
                try:
                    config[k] = float(v)
                except ValueError:
                    config[k] = v
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Feature map visualization for SALAD backbones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用预训练checkpoint提取特征响应图:
  python tools/visualize_feature_maps.py --image path/to/image.jpg --ckpt path/to/checkpoint.ckpt

  # 仅使用backbone（无checkpoint）:
  python tools/visualize_feature_maps.py --image path/to/image.jpg --backbone dinov2_vitb14

  # 指定backbone配置:
  python tools/visualize_feature_maps.py --image path/to/image.jpg --ckpt path/to/checkpoint.ckpt \\
    --bb-kv num_trainable_blocks=4 return_token=true

  # 指定aggregator配置:
  python tools/visualize_feature_maps.py --image path/to/image.jpg --ckpt path/to/checkpoint.ckpt \\
    --agg-kv num_channels=768 num_clusters=64

  # 调整热图透明度（更淡）:
  python tools/visualize_feature_maps.py --image path/to/image.jpg --ckpt path/to/checkpoint.ckpt --alpha 0.2

  # 调整热图亮度和透明度:
  python tools/visualize_feature_maps.py --image path/to/image.jpg --ckpt path/to/checkpoint.ckpt --alpha 0.2 --gamma 0.8
		"""
    )
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--out', default='./viz_out', help='Directory to save visualizations')
    parser.add_argument('--backbone', default='dinov2_vitb14',
                        help='Backbone name, e.g., resnet50 or dinov2_vitb14')
    parser.add_argument('--bb-kv', nargs='*', default=[],
                        help='Backbone config as key=value (e.g., return_token=true num_trainable_blocks=4)')
    parser.add_argument('--channels', type=int, default=16,
                        help='Number of channels to visualize in grid')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224],
                        help='Input image size (H W) for backbone')
    parser.add_argument('--cmap', default='jet', help='Matplotlib colormap')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to pretrained checkpoint file (if provided, will load weights from checkpoint)')
    parser.add_argument('--agg-arch', type=str, default='SALAD',
                        help='Aggregator architecture (only used when --ckpt is provided)')
    parser.add_argument('--agg-kv', nargs='*', default=[],
                        help='Aggregator config as key=value (e.g., num_channels=768 num_clusters=64)')
    parser.add_argument('--heatmap-only', action='store_true',
                        help='Only generate heatmap visualizations (skip channel grids)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Overlay transparency (0-1, smaller = lighter heatmap, default: 0.3)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma correction for heatmap intensity (<1 = brighter, >1 = darker, default: 1.0)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output image DPI (resolution, default: 300, higher = better quality but larger file)')
    parser.add_argument('--upscale', type=int, default=4,
                        help='Upscale factor for feature maps (default: 4, makes images 4x larger)')
    args = parser.parse_args()

    bb_config = parse_backbone_config_from_kv(args.bb_kv)
    agg_config = parse_backbone_config_from_kv(args.agg_kv) if args.agg_kv else None

    visualize_feature_maps(
        image_path=args.image,
        output_dir=args.out,
        backbone_arch=args.backbone,
        backbone_config=bb_config,
        num_channels_grid=args.channels,
        image_size=tuple(args.img_size),
        colormap=args.cmap,
        ckpt_path=args.ckpt,
        agg_arch=args.agg_arch,
        agg_config=agg_config,
        heatmap_only=args.heatmap_only,
        alpha=args.alpha,
        gamma=args.gamma,
        output_dpi=args.dpi,
        upscale_factor=args.upscale,
    )


if __name__ == '__main__':
    main()
