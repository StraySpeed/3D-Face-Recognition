import os
import time
from client import PointFaceClient
from server import PointFaceServer
from config import CONFIG
from logger import get_logger

def main():
    logger = get_logger("matching_simulation")
    logger.info("========== FHE 3D Face Matching Simulation ==========")

    # 1. 서버 구동을 위한 Context 준비 s
    context_path = CONFIG["PATH"]["secret.context"]
    if not os.path.exists(context_path):
        logger.error("Context 파일이 없습니다.")
        return

    # 2. 객체 초기화 (Client & Server)
    logger.info("1. 시스템 모듈 초기화 중...")
    client = PointFaceClient(context_path, logger.info)
    server = PointFaceServer(context_path, logger.info)

    # 서버에 암호화된 갤러리(DB) 로드
    gallery_dir = CONFIG["PATH"]["gallery_storage_enc"]
    is_loaded = server.load_gallery_individual_npy(load_dir=gallery_dir)
    if not is_loaded:
        logger.error("갤러리 로드 실패. 프로그램을 종료합니다.")
        return

    # 3. 테스트용 클라이언트 Query 데이터 설정
    test_identity = "id_0171" # 테스트할 갤러리 ID
    query_npy_path = os.path.join(CONFIG["PATH"]["gallery_dir"], test_identity, "001150_0171_F_NE_F.npy")

    # ====================================================================
    # [Simulation Start] 네트워크 통신 시나리오
    # ====================================================================
    logger.info(f"\n========== Matching Test for ID: {test_identity} ==========")
    
    total_start = time.time()
    
    # [Client] 1. 전처리 및 암호화
    enc_data = client.preprocess(query_npy_path)
    logger.info("[Network] Client -> Server: 암호화된 3D 스캔 데이터 전송")
    
    # [Server] 2. 암호문 매칭 추론 (enc_data 수신)
    logger.info("[Network] Server에서 동형암호 연산 시작...")
    enc_score_bytes = server.recognize_encrypted_id2(enc_data, id=test_identity)
    
    # [Client] 3. 결과 수신 및 복호화
    logger.info("[Network] Server -> Client: 암호화된 인증 결과 반환")
    is_authenticated = client.decrypt_result(enc_score_bytes)
    
    total_end = time.time()
    
    # ====================================================================
    # 결과 출력
    # ====================================================================
    logger.info("\n========== Final Result ==========")
    if is_authenticated:
        logger.info(f"인증 성공! (ID: {test_identity})")
    else:
        logger.info(f"인증 실패. (ID: {test_identity})")
        
    logger.info(f"Total time: {total_end - total_start:.4f} 초")

if __name__ == "__main__":
    main()