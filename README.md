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

또는 폴더 단위로 실행:
```bash
python flow.py \
  --image-dir "/path/to/플로우 테스트" \
  --brass-od-mm 250 \
  --mold-base-mm 100 \
  --out-dir ./out
```

## Windows에서 바로 실행 (질문 주신 케이스)
핵심: **`flow.py` 파일이 있는 폴더에서 실행**하거나, `flow.py`의 **절대경로**를 지정해야 합니다.

이미지 폴더가 예를 들어 아래라면:
`C:\Users\whddn\Desktop\2026 Business folders\콘크리트\플로우 테스트`

PowerShell에서:
```powershell
cd C:\작업\AI_concrete
python flow.py `
  --image-dir "C:\Users\whddn\Desktop\2026 Business folders\콘크리트\플로우 테스트" `
  --brass-od-mm 250 `
  --mold-base-mm 100 `
  --out-dir ".\out"
```

또는 현재 폴더가 어디든지 상관없이(절대경로 실행):
```powershell
python "C:\작업\AI_concrete\flow.py" `
  --image-dir "C:\Users\whddn\Desktop\2026 Business folders\콘크리트\플로우 테스트" `
  --brass-od-mm 250 `
  --mold-base-mm 100 `
  --out-dir "C:\작업\AI_concrete\out"
```

### 더 쉬운 방법 (권장)
저장소에 포함된 `run_flow.ps1` 스크립트를 사용하면, 스크립트가 자동으로 같은 폴더의 `flow.py`를 실행합니다.

```powershell
cd C:\작업\AI_concrete
.\run_flow.ps1 -ImageDir "C:\Users\whddn\Desktop\2026 Business folders\콘크리트\플로우 테스트" -OutDir ".\out" -Debug
```

문제가 있을 때(경로/실행 파일 확인):
```powershell
python .\flow.py `
  --image-dir "C:\Users\whddn\Desktop\2026 Business folders\콘크리트\플로우 테스트" `
  --brass-od-mm 250 `
  --mold-base-mm 100 `
  --out-dir ".\out" `
  --debug
```

실행 후 `.\out` 폴더에 아래 파일이 생성됩니다.
- `flow_annotated_1.png`, `flow_annotated_2.png`, ...
- `flow_test_annotated_results.png` (합성 결과)
- `flow_test_results.csv` (수치 결과)

## 입력/저장(JSON) 방식 추가
입력 데이터를 JSON으로 불러오고, 실행 결과 메타데이터를 JSON으로 저장할 수 있습니다.

입력 예시 (`input_config.json`):
```json
{
  "image_dir": "C:/Users/whddn/Desktop/2026 Business folders/콘크리트/플로우 테스트",
  "brass_od_mm": 250,
  "mold_base_mm": 100,
  "out_dir": "./out",
  "result_json": "./out/result_meta.json"
}
```

실행:
```powershell
python .\flow.py --input-json .\input_config.json
```

결과:
- 기존 출력(`flow_test_results.csv`, 주석 이미지, 합성 이미지)
- 추가 출력: `result_meta.json` (입력 이미지 목록, 결과 파일 경로, 행 개수)

## 안 될 때 빠른 점검
0. 현재 실행되는 `flow.py` 경로만 확인:
   ```powershell
   python .\flow.py --show-path
   ```
1. `python .\flow.py --debug ...`로 실행해서 `[debug] script=...` 경로가 현재 수정한 `flow.py`인지 확인
2. PowerShell 현재 경로 확인: `pwd`
3. 출력 폴더 확인: `dir .\out`
4. 필요한 라이브러리 설치:
   ```powershell
   pip install opencv-python numpy pandas matplotlib
   ```

## 테스트
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
