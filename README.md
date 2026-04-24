# AI_concrete

플로우 테이블(모르타르 확산) 이미지를 자동 분석하는 코드입니다.

## 이전 코드의 주요 문제
- 코드가 중복 붙여넣기되어 마지막에 문법 오류가 발생할 수 있었습니다.
- `display(df)`는 주피터 환경이 아니면 동작하지 않습니다.
- `cv2.imread()` 실패(None) 체크가 없어 런타임 오류 위치를 찾기 어려웠습니다.
- 합성 이미지 subplot이 `4x2`로 고정되어 이미지 개수가 바뀌면 깨질 수 있습니다.

## 개선 사항
- `flow.py`를 함수형 구조 + CLI 실행 구조로 정리했습니다.
- 이미지 로드 실패 시 파일 경로를 포함한 명확한 에러를 발생시킵니다.
- 황동 원(brass) 검출: Hough + HSV fallback으로 안정성을 높였습니다.
- 모르타르 경계 검출: LAB 기반 마스크와 중심/원형도 점수로 오검출을 줄였습니다.
- 합성 이미지 그리드를 입력 개수에 맞춰 자동 생성합니다.
- CSV/주석 이미지/합성 이미지를 저장합니다.

## 실행 방법
```bash
python flow.py \
  /path/img1.jpg /path/img2.jpg \
  --brass-od-mm 250 \
  --mold-base-mm 100 \
  --out-dir ./out
```

## 테스트
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
