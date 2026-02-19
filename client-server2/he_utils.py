import torch
import torch.nn as nn
import os, sys
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from model.pointface import PointFaceNet

class ClientBody(nn.Module):
    """
    PointNet++의 Feature Extraction 부분 (SA1 ~ SA4 + MaxPool)
    FC를 제외한 부분만
    입력: Point Cloud (Plain)
    출력: Global Feature (Plain, 512-dim)
    """
    def __init__(self, original_model):
        super(ClientBody, self).__init__()
        # test.py의 PointFaceEncoder 구조를 그대로 가져옴
        encoder = original_model.encoder
        self.sa1 = encoder.sa1
        self.sa2 = encoder.sa2
        self.sa3 = encoder.sa3
        self.sa4 = encoder.sa4
        
    def forward(self, x):
        xyz = x[:, :3, :]
        features = x
        
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Global Max Pooling
        global_feature = torch.max(l4_points, 2)[0]

        # FC를 서버로 넘김
        return global_feature

def merge_linear_bn(linear, bn):
    """
    FC Layer (Linear + Batchnorm)

    Linear Layer와 BatchNorm1d를 하나의 Linear Layer(W', b')로 병합하기 위함
    
    BN에서 weight와 bias를 반환

    :return: weight, bias
    """
    with torch.no_grad():
        # BN 파라미터
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        # Scale Factor: gamma / sqrt(var + eps)
        scale = gamma / torch.sqrt(var + eps)
        
        # Merged Weight: W * scale
        # Linear weight shape: (out, in) -> (512, 512)
        # scale shape: (512,) -> broadcast needed
        w_merged = linear.weight * scale.unsqueeze(1)
        
        # Merged Bias: (b - mean) * scale + beta
        # Linear bias가 없는 경우 0으로 처리
        linear_bias = linear.bias if linear.bias is not None else torch.zeros_like(mean)
        b_merged = (linear_bias - mean) * scale + beta
        #b_merged = torch.zeros_like(beta)

    return w_merged.t().numpy().tolist(), b_merged.numpy().tolist()

def load_server_weights(model_path, device='cpu'):
    """
    학습된 모델에서 FC Layer와 BN Layer를 추출하여 병합된 가중치를 반환
    """
    # 모델 로드
    full_model = PointFaceNet(num_classes=143) # num_classes는 dummy
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        full_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        full_model.load_state_dict(checkpoint)
    full_model.eval()
    
    # Encoder 내부의 fc (Linear + BN) 접근
    fc_layer = full_model.encoder.fc[0] # Linear
    bn_layer = full_model.encoder.fc[1] # BatchNorm1d
    
    # 병합
    W, b = merge_linear_bn(fc_layer, bn_layer)
    return W, b, full_model