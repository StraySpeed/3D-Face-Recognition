import os
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

from client import PointFaceClient
from server import PointFaceServer
from logger import get_logger
    
if __name__ == "__main__":
    # 1. 설정
    print = get_logger(name='client_server').info
    MODEL_PATH = "./checkpoints/pointface_epoch_200.pth" # 학습된 모델 경로
    GALLERY_DIR = "./dataset_matching/umbdb_unpreprocessed" # 등록할 얼굴들이 있는 폴더

    # 2. Client, Server 초기화
    client = PointFaceClient(MODEL_PATH, device='cpu', logger=print)
    server = PointFaceServer(client.get_public_context(), logger=print)

    # 3. 갤러리 암호화 등록
    # 여기서는 기존 DB를 이용함
    server.load_gallery_individual("./gallery_storage/umbdb_enc")
    # 기존 키를 그대로 가져옴
    client.set_public_context("./gallery_storage/umbdb_enc/secret.context")

    # 4. 인식 수행
    identities = sorted([d for d in os.listdir(GALLERY_DIR) if os.path.isdir(os.path.join(GALLERY_DIR, d)) and not d.startswith('.')])
    correct = 0; wrong = 0; total = 0
    for id in identities:
        person_dir = os.path.join(GALLERY_DIR, id)
        for f in os.listdir(person_dir):
            total += 1
            identity_file = os.path.join(person_dir, f)
            print(f"[Matching] Identity: {id}")

            # Client: 특징 추출 후 암호화해서 반환
            probe_data = client.get_embedding_and_encrypt(identity_file)
            
            # Server: 매칭 수행
            # 결과는 암호화된 (DotProduct, Norms) 리스트
            encrypted_score = server.recognize_encrypted_id(probe_data, id)
            
            # Client: 결과 복호화 및 판단
            # 통과 여부만을 제공함
            authenticaed = client.decrypt_result(encrypted_score)
            identity = id
            if not authenticaed:
                identity = "Unknown"

            print(f"[Result] Identity: {identity}")
            
            if authenticaed:
                correct += 1
            else:
                wrong += 1

    # 여기서 correct는 pass한 횟수를 의미
    print(f"[Result] Total : {total}, Correct : {correct} ({correct} / {total}), Wrong : {wrong} ({wrong} / {total})")