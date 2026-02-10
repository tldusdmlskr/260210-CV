import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ✅ 핵심: 통합 모듈 하나만 임포트하면 끝!
from upper_keypoint_model import UpperArchScorer

# 설정
IMG_DIR = "./images"
MODEL_PATH = "./models/best_model_upper_30.pth"
CSV_SAVE = "./results/final_scores_v2.csv"

def main():
    # 1. Scorer 초기화 (여기서 Gap 기준 설정 가능)
    scorer = UpperArchScorer(
        model_path=MODEL_PATH, 
        gap_threshold=20.0, # 20픽셀 이상이면 감점
        gap_penalty=5.0     # 하나당 5점 감점
    )
    
    # 2. 파일 리스트
    files = glob.glob(os.path.join(IMG_DIR, "*.png"))
    print(f"총 {len(files)}장 분석 시작...")
    
    results = []
    
    # 3. 루프 (매우 간단해짐)
    for i, f in enumerate(files):
        if i % 50 == 0: print(f"{i}...")
        
        # 분석 실행!
        res_dict, mask, img = scorer.process(f)
        results.append(res_dict)
        
    # 4. 저장
    df = pd.DataFrame(results)
    df.to_csv(CSV_SAVE, index=False)
    print("분석 완료!")

if __name__ == "__main__":
    main()