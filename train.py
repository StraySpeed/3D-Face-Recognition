"""
PointFace (model2) 학습 스크립트.

  - 모델: model2.PointFaceNet  (hid_ch 제거, squared-dist 제거)
  - feature_dim: 512
  - 체크포인트 디렉토리: checkpoints_facescape_v2
  - 체크포인트 파일명: pointface2_epoch_*.pth
  - 학습 전략: CE (ArcFace warmup) + SupConLoss
"""

import os
import time, datetime
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import PointFaceNet, DEFAULT_FEATURE_DIM
from model.loader import get_dataloader
from model.modules.arcface import ArcFaceHead
from model.modules.supcon import SupConLoss

from logger import get_logger
from config import CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes  = CONFIG["MODEL"]["num_classes"]
feature_dim  = DEFAULT_FEATURE_DIM  # 256
num_points   = CONFIG["MODEL"]["num_points"]

ARCFACE_S              = 32.0
SUPCON_LAMBDA          = 0.5
ARCFACE_MARGIN_MAX     = CONFIG["TRAIN"]["margin"]
ARCFACE_WARMUP_EPOCHS  = CONFIG["TRAIN"]["arcface_margin_warmup_epochs"]

supcon_criterion = SupConLoss(temperature=0.1)

model        = PointFaceNet(feature_dim=feature_dim).to(device)
arcface_head = ArcFaceHead(feature_dim, num_classes, s=ARCFACE_S, m=0.0).to(device)

optimizer = optim.Adam([
    {'params': model.parameters(),        'lr': CONFIG["TRAIN"]["learning_rate"]},
    {'params': arcface_head.parameters(), 'lr': CONFIG["TRAIN"]["learning_rate"]},
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CONFIG["TRAIN"]["epochs"],
    eta_min=1e-6,
)


def train_one_epoch(dataloader, epoch):
    model.train()
    arcface_head.train()
    total_loss = total_correct = total_samples = 0

    for batch_idx, (anchor, pos, labels) in enumerate(dataloader):
        anchor = anchor.to(device, non_blocking=True)
        pos    = pos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        emb_anchor   = model(anchor)
        emb_positive = model(pos)

        emb_all    = torch.cat([emb_anchor, emb_positive], dim=0)
        labels_all = torch.cat([labels, labels], dim=0)

        warmup_margin = min(ARCFACE_MARGIN_MAX,
                           ARCFACE_MARGIN_MAX * epoch / ARCFACE_WARMUP_EPOCHS)
        logits = arcface_head(emb_all, labels_all, margin=warmup_margin)
        loss_ce     = F.cross_entropy(logits, labels_all)
        loss_supcon = supcon_criterion(emb_all, labels_all)
        loss        = loss_ce + SUPCON_LAMBDA * loss_supcon

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(arcface_head.parameters()),
            max_norm=5.0,
        )
        optimizer.step()

        total_loss    += loss.item()
        total_correct += (logits.argmax(dim=1) == labels_all).sum().item()
        total_samples += labels_all.size(0)

        if batch_idx % 20 == 0:
            acc = total_correct / max(total_samples, 1) * 100
            print(f"Epoch [{epoch}] Batch [{batch_idx}]  "
                  f"Loss: {loss.item():.4f}  (CE: {loss_ce.item():.4f}, SupCon: {loss_supcon.item():.4f})  "
                  f"Acc: {acc:.2f}%")

    acc = total_correct / total_samples * 100
    return total_loss / len(dataloader), acc


def save_checkpoint(epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"pointface2_epoch_{epoch:03d}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict':     model.state_dict(),
        'arcface_state_dict':   arcface_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")


def resume_from_checkpoint(checkpoint_path, reset_optimizer=False):
    """
    True resume: optimizer + scheduler state까지 복원하여 원래 LR schedule을 그대로 이어감.

    옛 체크포인트(scheduler_state_dict 없음)는 last_epoch를 fast-forward해서
    cosine schedule상 같은 위치로 복원.
    """
    if not os.path.exists(checkpoint_path):
        return 0
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'arcface_state_dict' in ckpt:
        arcface_head.load_state_dict(ckpt['arcface_state_dict'])

    start_epoch = ckpt['epoch']

    if reset_optimizer:
        print(f"Resumed from epoch {start_epoch} (optimizer/scheduler reset).")
        return start_epoch

    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        sched_status = "scheduler restored"
    else:
        scheduler.last_epoch = start_epoch - 1
        scheduler.step()
        sched_status = f"scheduler fast-forwarded to epoch {start_epoch}"

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Resumed from epoch {start_epoch} (optimizer restored, {sched_status}, LR={current_lr:.6f}).")
    return start_epoch


if __name__ == '__main__':
    logger = get_logger(name='train2')
    print  = logger.info

    train_loader = get_dataloader(
        CONFIG["PATH"]["data_root"],
        batch_size=CONFIG["TRAIN"]["batch_size"],
        num_workers=CONFIG["TRAIN"]["num_workers"],
        train=True,
        num_points=num_points,
    )

    savepath  = CONFIG["PATH"]["checkpoint_dir"]
    max_epoch = CONFIG["TRAIN"]["epochs"]

    start_epoch = 0
    # 이어서 학습하려면 아래 두 줄 주석 해제 (true resume — LR schedule 그대로 이어감)
    CHECKPOINT_PATH = os.path.join(savepath, "pointface2_epoch_110.pth")
    start_epoch = resume_from_checkpoint(CHECKPOINT_PATH)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model2] params: {n_params/1e6:.3f}M, feature_dim={feature_dim}, num_points={num_points}")
    print(f"[model2] checkpoints: {savepath}")

    for epoch in range(start_epoch, max_epoch):
        start_time = time.time()
        avg_loss, train_acc = train_one_epoch(train_loader, epoch)
        scheduler.step()
        elapsed  = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=int(elapsed)))
        lr       = scheduler.get_last_lr()[0]

        print(f"[CE+SupCon] Epoch [{epoch}/{max_epoch}]  "
              f"Loss: {avg_loss:.4f}  TrainAcc: {train_acc:.2f}%  "
              f"Time: {time_str}  LR: {lr:.6f}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch + 1, savepath)
