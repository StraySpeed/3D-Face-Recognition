import re
import numpy as np
import matplotlib.pyplot as plt

def analyze_filtered_threshold(log_file_path):
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    scores = []

    pattern = r"Identity:\s*(?!Unknown)(?:id_\d+).*?Score:\s*([0-9\.]+)"
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            scores.append(float(match.group(1)))

    if not scores:
        return

    scores = np.array(scores)
    
    # 통계 출력
    print(f"\n=== Genuine Score Analysis (Filtered: {len(scores)} samples) ===")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Max:  {np.max(scores):.4f}")
    
    # 1. 통계 분석
    print(f"\n=== BioHashing Score Statistics (Hamming Distance) ===")
    print(f"Total Samples: {len(scores)}")
    print(f"Min Distance (Best Match): {np.min(scores):.4f}")
    print(f"Max Distance (Worst Match): {np.max(scores):.4f}")
    print(f"Mean Distance: {np.mean(scores):.4f}")
    
    # 2. 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color='green', edgecolor='black', alpha=0.7, label='Genuine Scores')
    plt.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.title('Hamming Distance Distribution (BioHashing)')
    plt.xlabel('Hamming Distance (0.0 = Same, 0.5 = Diff)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Hamming_Distance_Distribution.png")

    # 3. Threshold 계산
    
    th_95 = np.percentile(scores, 95) # 상위 5% 지점 (95% 통과)
    th_99 = np.percentile(scores, 99) # 상위 1% 지점 (99% 통과)
    
    print(f"\n=== Recommended Thresholds ===")
    print(f"Strict (FRR 5%): {th_95:.4f}")
    print(f"Loose  (FRR 1%): {th_99:.4f}")
    
    print(f">>> Recommended Threshold (FRR 1%): {th_99:.4f}")

# 실행
analyze_filtered_threshold("./logs/matching_biocode_log_20260217_012613.txt")