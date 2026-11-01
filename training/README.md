# 학습 체인

대회 기간에 실제로 돌린 연구용 스크립트를 그대로 옮겼습니다. 개인 환경의 절대경로만 환경 변수로 바꿨습니다. 그래서 그대로 실행하려면 아래의 원래 작업 폴더 구조를 다시 만들어야 합니다.

## 작업 폴더 구조

스크립트는 `SEOKCHEON_WS`(작업 폴더 루트) 아래에 다음 구조가 있다고 가정합니다.

```
$SEOKCHEON_WS/
  work/                    training/common/*.py, e2e.sh
                           + 대회 데이터 train.csv, vessel_catalog.csv
                           + 기반 패키지 폴더 (v27a, v27f, v27af 등)
  agents/round4/           training/hangul/*  (a8g.py는 agents/round4/hm/a8g.py)
                           training/latin/build_en2.sh, de_en2.py
                           logs/
  agents/latin-de/         training/latin/export_de.py, train14_ende.py, fit_en.py  (+ out/)
  agents/de-v2-full/       training/common/common_l.py, training/capcpu/
```

인터프리터는 환경 변수로 지정합니다.
- `PY`: CPU용 Python. numpy, pandas, lightgbm, rapidfuzz, unidecode가 필요합니다.
- `TPY`: 듀얼 인코더 학습용 Python. PyTorch(CUDA)가 필요합니다.

둘 다 지정하지 않으면 `python`을 씁니다.

## 한글 경로 (최종 v29q 레시피)

```bash
TOWER_D=96 TOWER_H=320 DE_SCRIPT=dualenc_v2dom7_gpu.py DOM_ALL=0 \
  bash agents/round4/build_ko5_gpu.sh v29q "kogen6, 200 epochs x 70k synthetic names, tower 96/320" \
  200 2 70000 v2d "$(seq -s, 2 2 198)" 2
bash agents/round4/check.sh v29q                            # 합성 한글 검증셋 + 영문 비교셋으로 패키지 실행
bash agents/round4/gate.sh v29q agents/round4/v29q/folds    # 실제 train 한글명 382척에 대한 폴드 밖 1위 적중률
```

`build_ko5_gpu.sh` 안에서 일어나는 일은 다음과 같습니다.

1. `dualenc_v2dom7_gpu.py`가 문자 듀얼 인코더 6개(폴드 5개 + 전체)를 하나씩 학습합니다.
   - 학습 쌍: `kogen6`이 만든 합성 한글 표기와 실제 train 쌍입니다.
   - 짝수 epoch마다 하드 네거티브를 다시 캡니다.
2. `export_de4.py`로 패키지용 가중치와 폴드별 가중치를 내보냅니다.
3. `train14_de.py`가 후보 쌍을 만듭니다.
   - train 행과 그 잡음 복사본(`ko_noise`, `ko_heavy`)을 실제 추론 코드(Predictor)에 통과시킵니다.
   - 이때 각 행이 학습에 쓰이지 않은 폴드의 인코더를 씁니다.
4. `domtrain.py build`가 합성 국내선 한글 쌍을 만듭니다.
5. `fit20.py`가 LightGBM LambdaRank 3개 시드를 학습합니다(rounds 600).

## 영문 경로 (v28g 레시피)

```bash
bash agents/round4/build_en2.sh v28g "Latin DE with hard-negative mining"   # 기본값: 15 epochs, reps 4, 합성 2만, 태그 C, 2 epoch마다 마이닝
```

이 명령은 다음 순서로 진행됩니다.

1. `de_en2.py`로 영문 듀얼 인코더 6개를 학습합니다.
2. `export_de.py`로 가중치를 내보냅니다.
3. `train14_ende.py`가 후보 쌍을 만듭니다.
   - 7가지 복사본을 씁니다: 원본, 잡음, 강한 잡음, 약어화 2회, 톤수 교란 원본, 톤수 교란 잡음.
   - 설정은 `P_TON_ERR=0.1`입니다.
4. `fit_en.py`로 순위 모델을 학습합니다.

GPU용 변형 `build_en2_gpu.sh`도 있었지만, v28g는 CPU 버전으로 만들었습니다.

## 최종 패키지 조립

- **v29v:** v28g 패키지의 `model/ko`를 v29q 결과로 바꾼 것입니다.
- **v36js:** v29v에 LLM 철자 판정기(`model/llm/judge.py` + Qwen3-1.7B)를 더한 것입니다. 설정은 `LLM_TAU=2.0`, `LLM_KAPPA=3.0`, `LLM_SUBSET_GUARD=True`입니다.
- 저장소의 `submission/`이 이 최종본입니다.

판정기 설정을 정한 사전 연구는 [`../llm_judge/`](../llm_judge/)에 있습니다.

## 포함하지 않은 것

- **기반 패키지:** 체인이 복사해 시작하는 이전 패키지(`work/v27a`, `work/v27f`, `work/v27af`)입니다. 대신 `submission/`을 기반으로 쓸 수 있습니다. 다만 체인 스크립트가 기반 패키지의 docstring 첫 줄을 확인(assert)하므로 그 부분을 고쳐야 합니다.
- **대회 데이터와 파생 파일:** 후보 쌍(`pairs_*.npz`), 합성 검증셋(`work/domtest`, `cmp9`), 이전 레시피의 쌍 묶음(`work/ko18`, `domtrain.py`의 `cv`/`final` 모드에서만 사용)이 해당합니다.
- **`loan_en.txt`:** 카탈로그 단어 중 영어 사전 단어 목록입니다. 없으면 `loanword.py`가 시스템 사전 `/usr/share/dict/web2`를 대신 씁니다.
