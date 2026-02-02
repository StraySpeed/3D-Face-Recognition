import os
import torch
from model.pointface import PointFaceNet

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

model = PointFaceNet(num_classes=...).to(device)

# Verification
def load_model_for_inference(model, checkpoint_path, device):
    """
    저장된 .pth 파일에서 가중치를 불러와 모델에 덮어씌움

    :param model: PointFace Model
    :param checkpoint_path: Checkpoint file path
    :param device: \"cuda / mps\" if `torch.cuda.is_available()` else \"cpu\"
    """
    if not os.path.exists(checkpoint_path):
        print(f"[DEBUG] File doesn't exist: {checkpoint_path}")
        return

    # 1. 파일 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 2. 모델의 가중치(state_dict)만 추출하여 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 그냥 모델 전체를 저장했다면 바로 로드
        model.load_state_dict(checkpoint)
        
    print(f"Weights loaded from {checkpoint_path}")
    
    # 3. 평가 모드로 전환 (Dropout, Batchnorm 고정)
    model.eval() 
    return model


if __name__ == '__main__':
    load_model_for_inference(model, "./checkpoints/pointface_epoch_010.pth", device)