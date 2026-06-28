import torch
from model.pointface import PointFaceNet
from config import CONFIG


def count_params(model):
    print(f"\n{'레이어':<45} {'파라미터':>12} {'학습가능':>10}")
    print("─" * 70)

    total = trainable = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total     += n
        trainable += n if param.requires_grad else 0
        print(f"  {name:<43} {n:>12,}  {'✓' if param.requires_grad else '✗':>10}")

    buf_total = 0
    print("\n─ Buffers (BN running stats 등, 비학습) ─")
    for name, buf in model.named_buffers():
        n = buf.numel()
        buf_total += n
        print(f"  {name:<43} {n:>12,}")

    print("─" * 70)

    # 모듈별 요약
    module_summary = {}
    for name, param in model.named_parameters():
        top = name.split('.')[0] + '.' + name.split('.')[1] if '.' in name else name
        module_summary[top] = module_summary.get(top, 0) + param.numel()

    print(f"\n{'[모듈별 요약]'}")
    print(f"  {'모듈':<40} {'파라미터':>12}")
    print("  " + "─" * 54)
    for mod, cnt in module_summary.items():
        print(f"  {mod:<40} {cnt:>12,}")

    print("  " + "─" * 54)
    print(f"  {'학습 가능 파라미터 합계':<40} {trainable:>12,}")
    print(f"  {'비학습 버퍼 합계':<40} {buf_total:>12,}")
    print(f"  {'전체 (Params + Buffers)':<40} {total + buf_total:>12,}")
    print(f"\n  모델 크기 추정 (float32): {(total + buf_total) * 4 / 1024 / 1024:.3f} MB")
    print(f"  추론 전용 (encoder, classifier 제외): ", end="")

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"{enc_params:,} params  ({enc_params * 4 / 1024 / 1024:.3f} MB)")

    return trainable, buf_total, total + buf_total


if __name__ == "__main__":
    model = PointFaceNet()
    count_params(model)
