# eval.py
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import os

from vpr_model import VPRModel
from utils.validation import get_validation_recalls
# Dataloader
from dataloaders.val.NordlandDataset import NordlandDataset
from dataloaders.val.MapillaryDataset import MSLS
from dataloaders.val.MapillaryTestDataset import MSLSTest
from dataloaders.val.PittsburghDataset import PittsburghDataset
from dataloaders.val.SPEDDataset import SPEDDataset

VAL_DATASETS = ['MSLS', 'MSLS_Test', 'pitts30k_test', 'pitts250k_test', 'Nordland', 'SPED']


# -------------------------------
# Flexible checkpoint loader
# -------------------------------
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


def input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406];
    STD = [0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])


def get_val_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)

    if 'nordland' in dataset_name:
        ds = NordlandDataset(input_transform=transform)

    elif 'msls_test' in dataset_name:
        ds = MSLSTest(input_transform=transform)

    elif 'msls' in dataset_name:
        ds = MSLS(input_transform=transform)

    elif 'pitts' in dataset_name:
        ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)

    elif 'sped' in dataset_name:
        ds = SPEDDataset(input_transform=transform)
    else:
        raise ValueError

    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth


def get_descriptors(model, dataloader, device, use_amp=True):
    descriptors = []
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (
                use_amp and device.type == 'cuda') else torch.cpu.amp.autocast(enabled=False)
    model.eval()
    with torch.no_grad():
        with amp_ctx:
            for batch in tqdm(dataloader, 'Calculating descriptors...'):
                imgs, _ = batch
                imgs = imgs.to(device, non_blocking=True)
                output = model(imgs).detach().cpu()
                descriptors.append(output)

    return torch.cat(descriptors, dim=0)


def load_model(ckpt_path, device, backbone_arch='dinov2_vits14'):
    """
    根据你当前配置构建 VPRModel，再用健壮加载器灌权重。
    注意：agg_config['num_channels'] 要与 backbone 输出通道一致：
      - dinov2_vitb14 通常是 768
      - dinov2_vitl14 通常是 1024/1152/1536 视实现而定
    """
    model = VPRModel(
        backbone_arch=backbone_arch,
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,  # 如果你的聚合头用的是 1536，这里要改成 1536
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
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
    print(f"Loaded model from {ckpt_path} successfully!")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters
    parser.add_argument("--ckpt_path", type=str, required=True, default=None, help="Path to the checkpoint")
    parser.add_argument("--backbone_arch", type=str, default='dinov2_vitb14', help="Backbone arch (must match ckpt)")
    parser.add_argument("--no_amp", action='store_true', help="Disable AMP (float16) inference on CUDA")

    # Datasets parameters
    parser.add_argument(
        '--val_datasets',
        nargs='+',
        default=VAL_DATASETS,
        help='Validation datasets to use',
        choices=VAL_DATASETS,
    )
    parser.add_argument('--image_size', nargs='*', default=None, help='Image size (int, tuple or None)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Num dataloader workers')
    parser.add_argument('--faiss_gpu', action='store_true', help='Use FAISS GPU for recall calc if supported')

    args = parser.parse_args()

    # Parse image size
    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (args.image_size[0], args.image_size[0])
        elif len(args.image_size) == 2:
            args.image_size = tuple(args.image_size)
        else:
            raise ValueError('Invalid image size, must be int, tuple or None')
        args.image_size = tuple(map(int, args.image_size))

    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.ckpt_path, device=device, backbone_arch=args.backbone_arch)

    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, args.image_size)
        val_loader = DataLoader(
            val_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0)
        )

        print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, device, use_amp=not args.no_amp)

        print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[:num_references]
        q_list = descriptors[num_references:]

        print('total_size', descriptors.shape[0], num_queries + num_references)

        testing = isinstance(val_dataset, MSLSTest)

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=args.faiss_gpu and (device.type == 'cuda'),
            testing=testing,
        )

        if testing:
            out_path = args.ckpt_path + '.' + model.agg_arch + '.preds.txt'
            val_dataset.save_predictions(preds, out_path)
            print(f"Saved predictions to {out_path}")

        del descriptors
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        print('========> DONE!\n')
