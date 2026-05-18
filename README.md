# Satellite SR MVP

위성/항공 영상 초해상도(Super Resolution) MVP 프로젝트입니다.

## 목표

저해상도 영상(LR)을 입력받아 고해상도 형태(SR)로 복원하고, Bicubic 결과와 비교합니다.

## MVP 파이프라인

1. 데이터 로딩
2. HR/LR 데이터 확인
3. 패치 생성
4. SR 모델 학습/추론
5. 결과 병합
6. Before / Bicubic / SR 비교

## 폴더 구조

data/
  raw/
    hr/
    lr/
  processed/
    hr_patches/
    lr_patches/
    semantic_masks/

scripts/
notebooks/
outputs/
models/

## 주의

기업 제공 데이터와 모델 가중치는 GitHub에 업로드하지 않습니다.
