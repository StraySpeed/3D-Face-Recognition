"""
measure_rank1_detail.py — 카테고리별 Rank-1 정확도 측정

갤러리: 중립 표정(neutral) 1개 per subject
프로브 카테고리:
  NU  중립 (갤러리 제외 추가 중립 파일)
  FE  표정 (비중립, 비가림)
  OC  가림 (occlusion, 임의 표정)
  PS  포즈 (해당 없으면 건너뜀)

지원 데이터셋:
  UMBDB     → NU, FE, OC   (PS 없음)
  FaceScape → FE only       (중립이 1개뿐 → NU 없음, OC/PS 없음)

사용법:
  python measure_rank1_detail.py                        # 두 데이터셋 모두
  python measure_rank1_detail.py --dataset umbdb
  python measure_rank1_detail.py --dataset facescape
  python measure_rank1_detail.py --ckpt_umbdb checkpoints7_umbdb/pointface_epoch_300.pth
"""

import argparse
import os
import glob

import torch
import numpy as np

from config import CONFIG


# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────

def _latest_checkpoint(ckpt_dir: str) -> str:
    files = glob.glob(os.path.join(ckpt_dir, "pointface_epoch_200.pth"))
    if not files:
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_dir}")
    return max(files, key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))


# ──────────────────────────────────────────────────────────────
# 파일명 → 카테고리 분류
# ──────────────────────────────────────────────────────────────

def _cat_umbdb(fname: str) -> str:
    """
    파일명 형식: <acq>_<subj>_<gender>_<expr>_<occ>.npy
    반환: 'NE_F' | 'FE' | 'OC'
    """
    stem  = os.path.splitext(fname)[0]
    parts = stem.split('_')
    expr  = parts[-2]   # NE / SM / BO / AN / MI / OM / DI
    occ   = parts[-1]   # F / O
    if occ == 'O':
        return 'OC'
    if expr == 'NE':
        return 'NE_F'   # 갤러리 후보 또는 NU 프로브
    return 'FE'


def _cat_facescape(fname: str) -> str:
    """
    파일명 형식: <num>_<name>.npy
    1_neutral → 'GALLERY', 그 외 → 'FE'
    """
    try:
        num = int(fname.split('_')[0])
    except ValueError:
        return 'UNKNOWN'
    return 'GALLERY' if num == 1 else 'FE'


# ──────────────────────────────────────────────────────────────
# 모델 로더 (model2 → model 순서로 시도)
# ──────────────────────────────────────────────────────────────

def _load_model(model_path: str, feature_dim: int, num_points: int, device: torch.device):
    for module_name in ('model2', 'model.pointface'):
        try:
            if module_name == 'model2':
                from model import PointFaceNet
            else:
                from model.pointface import PointFaceNet

            net = PointFaceNet(feature_dim=feature_dim).to(device)
            ckpt = torch.load(model_path, map_location=device)
            net.load_state_dict(ckpt.get('model_state_dict', ckpt))
            net.eval()
            print(f"  [{module_name}] 로드 완료: {os.path.basename(model_path)}")
            return net, PointFaceNet
        except Exception:
            continue
    raise RuntimeError(f"model / model2 모두 로드 실패: {model_path}")


# ──────────────────────────────────────────────────────────────
# 임베딩 추출 & Rank-1 계산
# ──────────────────────────────────────────────────────────────

def _embed(net, NetClass, npy_path: str, num_points: int, device: torch.device):
    if not os.path.exists(npy_path):
        return None
    try:
        points = np.load(npy_path)[:, :3]
        tensor = NetClass.preprocess(points, num_points=num_points, device=str(device))
        with torch.no_grad():
            emb = net(tensor).cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-8 else emb
    except Exception as e:
        print(f"  [warn] {os.path.basename(npy_path)}: {e}")
        return None


def _rank1(probe_embs, probe_ids, gallery_embs, gallery_ids):
    """코사인 유사도 기반 Rank-1 정확도."""
    if not probe_embs:
        return None, 0, 0
    P  = np.array(probe_embs)
    G  = np.array(gallery_embs)
    pl = np.array(probe_ids)
    gl = np.array(gallery_ids)
    pred    = gl[np.argmax(P @ G.T, axis=1)]
    correct = int(np.sum(pred == pl))
    total   = len(pl)
    return correct / total * 100, correct, total


def _print_row(tag: str, acc, correct: int, total: int, skip_msg: str = ''):
    if skip_msg:
        print(f"  {tag:<4s}  --        {skip_msg}")
    elif acc is None:
        print(f"  {tag:<4s}  --        (프로브 없음)")
    else:
        print(f"  {tag:<4s}  {acc:6.2f}%   ({correct}/{total})")


# ──────────────────────────────────────────────────────────────
# UMBDB 평가
# ──────────────────────────────────────────────────────────────

def evaluate_umbdb(model_path: str, data_root: str,
                   feature_dim: int, num_points: int, device: torch.device):
    print(f"\n{'='*58}")
    print(f"  UMBDB — 카테고리별 Rank-1")
    print(f"  데이터: {data_root}")
    print(f"{'='*58}")

    net, NetClass = _load_model(model_path, feature_dim, num_points, device)

    identities = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
    )

    gallery_embs, gallery_ids = [], []
    # NU: 갤러리 제외 추가 중립(NE_F) 파일
    # FE: 비중립 + 비가림
    # OC: 가림 전체
    probes: dict[str, tuple[list, list]] = {
        'NU': ([], []),
        'FE': ([], []),
        'OC': ([], []),
    }

    print(f"  임베딩 추출 중 ({len(identities)} subjects)...")
    for idx, ident in enumerate(identities):
        d     = os.path.join(data_root, ident)
        files = sorted(f for f in os.listdir(d) if f.endswith('.npy'))

        ne_f_files = [f for f in files if _cat_umbdb(f) == 'NE_F']
        fe_files   = [f for f in files if _cat_umbdb(f) == 'FE']
        oc_files   = [f for f in files if _cat_umbdb(f) == 'OC']

        if not ne_f_files:
            continue

        # 갤러리: NE_F 중 첫 번째
        gal_emb = _embed(net, NetClass, os.path.join(d, ne_f_files[0]),
                         num_points, device)
        if gal_emb is None:
            continue
        gallery_embs.append(gal_emb)
        gallery_ids.append(idx)

        # NU 프로브: 나머지 NE_F
        for f in ne_f_files[1:]:
            emb = _embed(net, NetClass, os.path.join(d, f), num_points, device)
            if emb is not None:
                probes['NU'][0].append(emb)
                probes['NU'][1].append(idx)

        # FE 프로브
        for f in fe_files:
            emb = _embed(net, NetClass, os.path.join(d, f), num_points, device)
            if emb is not None:
                probes['FE'][0].append(emb)
                probes['FE'][1].append(idx)

        # OC 프로브
        for f in oc_files:
            emb = _embed(net, NetClass, os.path.join(d, f), num_points, device)
            if emb is not None:
                probes['OC'][0].append(emb)
                probes['OC'][1].append(idx)

    print(f"\n  갤러리: {len(gallery_embs)}개  (NE_F × 1 per subject)")
    print(f"  {'카테고리':<4s}  {'Rank-1':>7s}   (정답/전체)")
    print(f"  {'-'*38}")

    for cat in ['NU', 'FE', 'OC']:
        embs, ids = probes[cat]
        acc, correct, total = _rank1(embs, ids, gallery_embs, gallery_ids)
        _print_row(cat, acc, correct, total)

    _print_row('PS', None, 0, 0, skip_msg='UMBDB에 포즈 데이터 없음')
    print()


# ──────────────────────────────────────────────────────────────
# FaceScape 평가
# ──────────────────────────────────────────────────────────────

def evaluate_facescape(model_path: str, data_root: str,
                       feature_dim: int, num_points: int, device: torch.device):
    print(f"\n{'='*58}")
    print(f"  FaceScape — 카테고리별 Rank-1")
    print(f"  데이터: {data_root}")
    print(f"{'='*58}")

    net, NetClass = _load_model(model_path, feature_dim, num_points, device)

    def _sort_key(x):
        return int(x) if x.isdigit() else x

    identities = sorted(
        (d for d in os.listdir(data_root)
         if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')),
        key=_sort_key
    )

    gallery_embs, gallery_ids = [], []
    fe_embs, fe_ids           = [], []

    print(f"  임베딩 추출 중 ({len(identities)} subjects)...")
    for idx, ident in enumerate(identities):
        d     = os.path.join(data_root, ident)
        files = [f for f in os.listdir(d) if f.endswith('.npy')]

        gal_file   = next((f for f in files if _cat_facescape(f) == 'GALLERY'), None)
        expr_files = [f for f in files if _cat_facescape(f) == 'FE']

        if gal_file is None:
            continue
        gal_emb = _embed(net, NetClass, os.path.join(d, gal_file),
                         num_points, device)
        if gal_emb is None:
            continue
        gallery_embs.append(gal_emb)
        gallery_ids.append(idx)

        for f in expr_files:
            emb = _embed(net, NetClass, os.path.join(d, f), num_points, device)
            if emb is not None:
                fe_embs.append(emb)
                fe_ids.append(idx)

    print(f"\n  갤러리: {len(gallery_embs)}개  (1_neutral.npy per subject)")
    print(f"  {'카테고리':<4s}  {'Rank-1':>7s}   (정답/전체)")
    print(f"  {'-'*38}")

    _print_row('NU', None, 0, 0, skip_msg='neutral 1개 → 갤러리로 사용, 프로브 없음')

    acc, correct, total = _rank1(fe_embs, fe_ids, gallery_embs, gallery_ids)
    _print_row('FE', acc, correct, total)

    _print_row('OC', None, 0, 0, skip_msg='FaceScape에 가림 데이터 없음')
    _print_row('PS', None, 0, 0, skip_msg='FaceScape에 포즈 데이터 없음')
    print()


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='카테고리별(NU/FE/OC/PS) Rank-1 정확도 측정'
    )
    parser.add_argument(
        '--dataset', choices=['umbdb', 'facescape', 'all'], default='all',
        help='평가할 데이터셋 (기본: all)'
    )
    parser.add_argument('--ckpt_umbdb',     default=None,
                        help='UMBDB 체크포인트 경로 (미지정 시 최신 자동 선택)')
    parser.add_argument('--ckpt_facescape', default=None,
                        help='FaceScape 체크포인트 경로 (미지정 시 최신 자동 선택)')
    parser.add_argument('--feature_dim',    type=int,
                        default=CONFIG["MODEL"]["feature_dim"])
    parser.add_argument('--num_points',     type=int,
                        default=CONFIG["MODEL"]["num_points"])
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_args()

    BASE = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(args.device)

    UMBDB_DATA      = os.path.join(BASE, 'dataset_matching', 'umbdb_unpreprocessed')
    FACESCAPE_DATA  = os.path.join(BASE, 'dataset_matching', 'facescape')
    UMBDB_CKPT_DIR  = os.path.join(BASE, 'checkpoints8_umbdb')
    FS_CKPT_DIR     = os.path.join(BASE, 'checkpoints8_facescape')

    if args.dataset in ('umbdb', 'all'):
        ckpt = args.ckpt_umbdb or _latest_checkpoint(UMBDB_CKPT_DIR)
        evaluate_umbdb(ckpt, UMBDB_DATA, args.feature_dim, args.num_points, device)

    if args.dataset in ('facescape', 'all'):
        ckpt = args.ckpt_facescape or _latest_checkpoint(FS_CKPT_DIR)
        evaluate_facescape(ckpt, FACESCAPE_DATA, args.feature_dim, args.num_points, device)


if __name__ == '__main__':
    main()
