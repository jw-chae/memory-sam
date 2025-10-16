# Changelog

## [Unreleased] - 2025-10-16

### Added
- **손상된 이미지 자동 스킵 기능**
  - 손상된 이미지 파일(truncated, corrupted) 자동 감지 및 스킵
  - 스킵된 파일 목록을 결과 요약에 표시
  - OSError, IOError 등 이미지 로딩 오류 처리
  
### Changed
- `memory_sam_predictor.py`: 이미지 로드 시 에러 처리 강화
  - `pil_image.load()` 호출로 손상된 파일 조기 감지
  - 에러 발생 시 에러 정보를 포함한 딕셔너리 반환
  
- `memory_sam_ui.py`: 스킵된 파일 추적 및 표시
  - `skipped_files` 리스트로 스킵된 파일 추적
  - 처리 결과에 스킵된 파일 수와 이름 표시
  - 경고 메시지에 파일명 명시

### Technical Details
```python
# Before: 손상된 파일 시 전체 프로세스 중단
image = np.array(Image.open(image_path).convert("RGB"))
# OSError 발생 → 프로그램 크래시

# After: 손상된 파일 스킵하고 계속 진행
try:
    pil_image = Image.open(image_path)
    pil_image.load()  # Force load to detect corruption
    image = np.array(pil_image.convert("RGB"))
except (OSError, IOError) as e:
    return {"error": f"이미지 파일 손상: {e}", ...}
    # 다음 파일로 계속
```

### User Experience Improvements
- **처리 중단 방지**: 손상된 파일이 있어도 전체 프로세스 계속 진행
- **명확한 피드백**: 어떤 파일이 스킵되었는지 명시적으로 표시
- **요약 정보**: 처리 완료 후 스킵된 파일 통계 제공

## [Previous] - 2025-10-16

### Performance Optimization (7x faster)
- DINOv3 크기 최적화로 7배 속도 향상 (30s → 4s per image)
- 90% 디스크 공간 절약
- 메모리 누수 수정
- 다양한 버그 수정

---

**Date**: 2025-10-16  
**Version**: 1.1-dev

