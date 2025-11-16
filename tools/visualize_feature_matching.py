import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径，使脚本可以从任何位置运行
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.resolve()

# 确保项目根目录在路径中
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 如果从项目根目录运行，也添加当前目录
cwd = Path(os.getcwd()).resolve()
if cwd != project_root and str(cwd) not in sys.path:
    # 检查当前目录是否包含vpr_model.py
    if (cwd / 'vpr_model.py').exists():
        sys.path.insert(0, str(cwd))

# 验证关键文件是否存在
vpr_model_path = project_root / 'vpr_model.py'
if not vpr_model_path.exists():
    # 尝试从当前工作目录查找
    cwd_vpr = Path(os.getcwd()) / 'vpr_model.py'
    if cwd_vpr.exists():
        if str(Path(os.getcwd()).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(os.getcwd()).resolve()))
    else:
        raise FileNotFoundError(
            f"Cannot find vpr_model.py.\n"
            f"Tried: {vpr_model_path}\n"
            f"Tried: {cwd_vpr}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Script location: {script_dir}\n"
            f"Project root: {project_root}\n"
            f"Please run from project root or ensure vpr_model.py exists."
        )

# 调试信息（可选）
if os.environ.get('DEBUG_IMPORTS', ''):
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}")
    print(f"vpr_model.py exists: {vpr_model_path.exists()}")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist

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


def load_ckpt_flex(model, ckpt_path,
                   strip_prefixes=('state_dict.', 'model.', 'module.'),
                   replace_prefixes=(('backbone.model.', 'backbone.'),
                                     ('aggregator.', 'aggregator.')),
                   verbose=True):
    """加载checkpoint的灵活函数"""
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
    for k, v in new_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v

    model.load_state_dict(filtered, strict=False)
    return model


def load_pretrained_model(ckpt_path: str, device: str,
                          backbone_arch: str = 'dinov2_vitb14',
                          backbone_config: dict | None = None,
                          agg_arch: str = 'SALAD',
                          agg_config: dict | None = None):
    """从checkpoint加载预训练模型"""
    if backbone_config is None:
        backbone_config = {
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        }

    if agg_config is None:
        if 'dinov2_vitb' in backbone_arch.lower():
            num_channels = 768
        elif 'dinov2_vitl' in backbone_arch.lower():
            num_channels = 1024
        elif 'dinov2_vitg' in backbone_arch.lower():
            num_channels = 1536
        elif 'dinov2_vits' in backbone_arch.lower():
            num_channels = 384
        else:
            num_channels = 768

        agg_config = {
            'num_channels': num_channels,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        }

    model = VPRModel(
        backbone_arch=backbone_arch,
        backbone_config=backbone_config,
        agg_arch=agg_arch,
        agg_config=agg_config,
    ).to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_ckpt_flex(
        model,
        ckpt_path,
        strip_prefixes=('state_dict.', 'model.', 'module.'),
        replace_prefixes=(('backbone.model.', 'backbone.'), ('aggregator.', 'aggregator.')),
        verbose=True
    )

    model.eval()
    return model


def extract_features(backbone, image: torch.Tensor):
    """提取特征图"""
    with torch.no_grad():
        out = backbone(image)

    if isinstance(out, (tuple, list)):
        features = out[0]
    else:
        features = out

    return features  # [1, C, H, W]


def compute_feature_similarity(feat1: torch.Tensor, feat2: torch.Tensor):
    """
    计算两个特征图之间的相似度矩阵

    Args:
        feat1: [1, C, H1, W1]
        feat2: [1, C, H2, W2]

    Returns:
        similarity: [H1*W1, H2*W2] 相似度矩阵
    """
    B, C, H1, W1 = feat1.shape
    B, C, H2, W2 = feat2.shape

    # 展平空间维度
    f1 = feat1.reshape(B, C, H1 * W1).permute(0, 2, 1)  # [1, H1*W1, C]
    f2 = feat2.reshape(B, C, H2 * W2).permute(0, 2, 1)  # [1, H2*W2, C]

    # L2归一化
    f1 = F.normalize(f1, p=2, dim=2)
    f2 = F.normalize(f2, p=2, dim=2)

    # 计算余弦相似度
    similarity = torch.bmm(f1, f2.transpose(1, 2))  # [1, H1*W1, H2*W2]
    similarity = similarity.squeeze(0)  # [H1*W1, H2*W2]

    return similarity


def find_matches(similarity: torch.Tensor, top_k: int = 50, threshold: float = 0.5):
    """
    找到最佳匹配点对

    Args:
        similarity: [H1*W1, H2*W2] 相似度矩阵
        top_k: 返回前k个匹配
        threshold: 相似度阈值

    Returns:
        matches: list of (idx1, idx2, score) tuples
    """
    H1W1, H2W2 = similarity.shape

    # 双向匹配：从feat1到feat2和从feat2到feat1
    matches_12 = []  # feat1 -> feat2
    matches_21 = []  # feat2 -> feat1

    # feat1 -> feat2: 对每个feat1的位置，找feat2中最佳匹配
    for i in range(H1W1):
        best_j = similarity[i].argmax().item()
        score = similarity[i, best_j].item()
        if score >= threshold:
            matches_12.append((i, best_j, score))

    # feat2 -> feat1: 对每个feat2的位置，找feat1中最佳匹配
    for j in range(H2W2):
        best_i = similarity[:, j].argmax().item()
        score = similarity[best_i, j].item()
        if score >= threshold:
            matches_21.append((best_i, j, score))

    # 只保留双向匹配（mutual nearest neighbors）
    mutual_matches = []
    for i, j, s12 in matches_12:
        # 检查是否存在反向匹配
        if any(m[1] == i and m[0] == j for m in matches_21):
            mutual_matches.append((i, j, s12))

    # 按相似度排序，取top_k
    mutual_matches.sort(key=lambda x: x[2], reverse=True)
    return mutual_matches[:top_k]


def visualize_feature_matching(
        query_path: str,
        reference_path: str,
        output_dir: str,
        backbone_arch: str = 'dinov2_vitb14',
        backbone_config: dict | None = None,
        image_size=(224, 224),
        ckpt_path: str | None = None,
        agg_arch: str = 'SALAD',
        agg_config: dict | None = None,
        top_k: int = 50,
        threshold: float = 0.5,
        output_dpi: int = 300,
        upscale_factor: int = 4
):
    """
    可视化两幅图像之间的特征匹配

    Args:
        query_path: 查询图像路径
        reference_path: 参考图像路径
        output_dir: 输出目录
        backbone_arch: backbone架构
        backbone_config: backbone配置
        image_size: 输入图像尺寸
        ckpt_path: checkpoint路径
        agg_arch: aggregator架构
        agg_config: aggregator配置
        top_k: 显示前k个匹配
        threshold: 相似度阈值
        output_dpi: 输出DPI
        upscale_factor: 放大倍数
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Output directory: {output_dir}")

    if backbone_config is None:
        backbone_config = {}

    # 加载模型
    if ckpt_path is not None:
        print(f"Loading pretrained weights from: {ckpt_path}")
        model = load_pretrained_model(
            ckpt_path=ckpt_path,
            device=device,
            backbone_arch=backbone_arch,
            backbone_config=backbone_config,
            agg_arch=agg_arch,
            agg_config=agg_config
        )
        backbone = model.backbone
    else:
        print("Using backbone without checkpoint")
        backbone = helper.get_backbone(backbone_arch, backbone_config).to(device).eval()

    # 加载图像
    query_img = load_image(query_path, image_size=image_size).to(device)
    ref_img = load_image(reference_path, image_size=image_size).to(device)

    # 提取特征
    print("Extracting features...")
    feat_query = extract_features(backbone, query_img)
    feat_ref = extract_features(backbone, ref_img)

    B, C, Hq, Wq = feat_query.shape
    B, C, Hr, Wr = feat_ref.shape

    print(f"Query feature shape: {C} channels, {Hq}x{Wq} spatial")
    print(f"Reference feature shape: {C} channels, {Hr}x{Wr} spatial")

    # 计算相似度矩阵
    print("Computing similarity matrix...")
    similarity = compute_feature_similarity(feat_query, feat_ref)

    # 找到匹配点
    print(f"Finding matches (top_k={top_k}, threshold={threshold})...")
    matches = find_matches(similarity, top_k=top_k, threshold=threshold)
    print(f"Found {len(matches)} mutual matches")

    # 加载原始图像用于可视化
    img_query = Image.open(query_path).convert('RGB')
    img_ref = Image.open(reference_path).convert('RGB')

    # 调整图像大小以匹配特征图空间尺寸
    img_query_resized = img_query.resize((Wq * upscale_factor, Hq * upscale_factor), Image.Resampling.LANCZOS)
    img_ref_resized = img_ref.resize((Wr * upscale_factor, Hr * upscale_factor), Image.Resampling.LANCZOS)

    # 1. 可视化匹配点连线
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(img_query_resized)
    axes[0].set_title('Query Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img_ref_resized)
    axes[1].set_title('Reference Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 绘制匹配线
    for idx_q, idx_r, score in matches:
        # 将索引转换为坐标
        y_q, x_q = divmod(idx_q, Wq)
        y_r, x_r = divmod(idx_r, Wr)

        # 放大坐标
        x_q_scaled = (x_q + 0.5) * upscale_factor
        y_q_scaled = (y_q + 0.5) * upscale_factor
        x_r_scaled = (x_r + 0.5) * upscale_factor
        y_r_scaled = (y_r + 0.5) * upscale_factor

        # 计算颜色（根据相似度）
        color = plt.cm.viridis(score)

        # 绘制点
        axes[0].plot(x_q_scaled, y_q_scaled, 'o', color=color, markersize=8, alpha=0.8)
        axes[1].plot(x_r_scaled, y_r_scaled, 'o', color=color, markersize=8, alpha=0.8)

        # 绘制连线（需要调整坐标，因为两张图并排）
        # 计算连接线的坐标
        x1, y1 = x_q_scaled, y_q_scaled
        x2 = img_query_resized.width + x_r_scaled
        y2 = y_r_scaled

        # 创建连接线
        line = mpatches.ConnectionPatch(
            (x1, y1), (x2, y2),
            "data", "data",
            axesA=axes[0], axesB=axes[1],
            color=color, linewidth=1.5, alpha=0.6, linestyle='-'
        )
        fig.add_artist(line)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_matching_lines.png'), dpi=output_dpi, bbox_inches='tight')
    plt.close(fig)

    # 2. 可视化相似度矩阵
    similarity_np = to_numpy(similarity)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_np, cmap='viridis', aspect='auto')
    ax.set_xlabel('Reference Image Spatial Positions', fontsize=12)
    ax.set_ylabel('Query Image Spatial Positions', fontsize=12)
    ax.set_title('Feature Similarity Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'similarity_matrix.png'), dpi=output_dpi, bbox_inches='tight')
    plt.close(fig)

    # 3. 可视化特征热图对比
    feat_q_np = to_numpy(feat_query[0])  # [C, Hq, Wq]
    feat_r_np = to_numpy(feat_ref[0])  # [C, Hr, Wr]

    # 平均激活
    avg_q = feat_q_np.mean(axis=0)
    avg_r = feat_r_np.mean(axis=0)
    avg_q = normalize_minmax(avg_q)
    avg_r = normalize_minmax(avg_r)

    # 放大特征图
    avg_q_large = zoom(avg_q, (upscale_factor, upscale_factor), order=1)
    avg_r_large = zoom(avg_r, (upscale_factor, upscale_factor), order=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Query图像和热图
    axes[0, 0].imshow(img_query_resized)
    axes[0, 0].set_title('Query Image', fontsize=12)
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(avg_q_large, cmap='jet', aspect='auto')
    axes[0, 1].set_title('Query Feature Heatmap', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Reference图像和热图
    axes[1, 0].imshow(img_ref_resized)
    axes[1, 0].set_title('Reference Image', fontsize=12)
    axes[1, 0].axis('off')

    im2 = axes[1, 1].imshow(avg_r_large, cmap='jet', aspect='auto')
    axes[1, 1].set_title('Reference Feature Heatmap', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_heatmap_comparison.png'), dpi=output_dpi, bbox_inches='tight')
    plt.close(fig)

    # 4. 保存匹配统计信息
    with open(os.path.join(output_dir, 'matching_stats.txt'), 'w') as f:
        f.write(f"Feature Matching Statistics\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Query image: {query_path}\n")
        f.write(f"Reference image: {reference_path}\n")
        f.write(f"Query feature shape: {C} channels, {Hq}x{Wq} spatial\n")
        f.write(f"Reference feature shape: {C} channels, {Hr}x{Wr} spatial\n\n")
        f.write(f"Total matches found: {len(matches)}\n")
        f.write(f"Top-K: {top_k}\n")
        f.write(f"Similarity threshold: {threshold}\n\n")
        if matches:
            f.write(f"Average similarity: {np.mean([m[2] for m in matches]):.4f}\n")
            f.write(f"Max similarity: {max([m[2] for m in matches]):.4f}\n")
            f.write(f"Min similarity: {min([m[2] for m in matches]):.4f}\n")

    print(f'\n✓ Saved visualizations to: {output_dir}')
    print(f'  - feature_matching_lines.png (匹配点连线图)')
    print(f'  - similarity_matrix.png (相似度矩阵)')
    print(f'  - feature_heatmap_comparison.png (特征热图对比)')
    print(f'  - matching_stats.txt (匹配统计信息)')


def parse_backbone_config_from_kv(args_kv: list[str]) -> dict:
    """从key=value格式解析配置"""
    config: dict = {}
    for kv in args_kv:
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        k, v = k.strip(), v.strip()
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
        description='Visualize feature matching between two images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用:
  python tools/visualize_feature_matching.py --query img1.jpg --ref img2.jpg --ckpt weights/model.ckpt

  # 指定匹配数量:
  python tools/visualize_feature_matching.py --query img1.jpg --ref img2.jpg --ckpt weights/model.ckpt --top-k 100

  # 调整相似度阈值:
  python tools/visualize_feature_matching.py --query img1.jpg --ref img2.jpg --ckpt weights/model.ckpt --threshold 0.6
		"""
    )
    parser.add_argument('--query', required=True, help='Path to query image')
    parser.add_argument('--ref', required=True, help='Path to reference image')
    parser.add_argument('--out', default='./viz_matching', help='Output directory')
    parser.add_argument('--backbone', default='dinov2_vitb14', help='Backbone architecture')
    parser.add_argument('--bb-kv', nargs='*', default=[], help='Backbone config as key=value')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224], help='Input image size')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--agg-arch', type=str, default='SALAD', help='Aggregator architecture')
    parser.add_argument('--agg-kv', nargs='*', default=[], help='Aggregator config as key=value')
    parser.add_argument('--top-k', type=int, default=50, help='Number of top matches to visualize')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold for matches')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--upscale', type=int, default=4, help='Upscale factor for visualization')
    args = parser.parse_args()

    bb_config = parse_backbone_config_from_kv(args.bb_kv)
    agg_config = parse_backbone_config_from_kv(args.agg_kv) if args.agg_kv else None

    visualize_feature_matching(
        query_path=args.query,
        reference_path=args.ref,
        output_dir=args.out,
        backbone_arch=args.backbone,
        backbone_config=bb_config,
        image_size=tuple(args.img_size),
        ckpt_path=args.ckpt,
        agg_arch=args.agg_arch,
        agg_config=agg_config,
        top_k=args.top_k,
        threshold=args.threshold,
        output_dpi=args.dpi,
        upscale_factor=args.upscale,
    )


if __name__ == '__main__':
    main()

