diff --git a/README.md b/README.md
old mode 100755
new mode 100644
index f7fc2e9d0c3e41ec3d2980860829686c0a0a5894..4b511fa5a982dc81138e159ed1dddb159bfd816d
--- a/README.md
+++ b/README.md
@@ -1,41 +1,113 @@
 # 결함 유발 코드 예측의 근거를 설명하는 딥러닝 기반 디버깅 워크벤치 개발
-본 문서는 `(202300000001212) 결함 유발 코드 예측의 근거를 설명하는 딥러닝 기반 디버깅 워크벤치 개발` 연구과제의 산출물을 정리한 것입니다. 
-
-## 1차년도 (2023.03.01~ 2024.02.29)
-
-각 모듈을 클릭하면 해당 파일로 이동할 수 있습니다.
-
-| 모듈명                            | 설명 |
-|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
-| **Datasets**                      |                                                                                                                                                                                                                                                                                                                                                                                                         |
-| [`sw_metrics/`](./Datasets/sw_metrics/README.md)    | 9종류의 오픈소스 프로젝트 및 JIRA 이슈 리포트로부터 추출된 소프트웨어 결함 메트릭 데이터셋 (SDP)                                                                                                                                                                                                                                                                                                      |
-| [`kamei_metrics/`](./Datasets/kamei_metrics/README.md) | 13종류의 오픈소스 프로젝트 및 JIRA 이슈 리포트로부터 추출된 commit 메타 데이터 기반의 Just-In-Time 소프트웨어 결함 데이터셋 (JIT_SDP)                                                                                                                                                                                                                                                                    |
-| [`cogc_metrics/`](./Datasets/cogc_metrics/README.md)  | 13종류의 오픈소스 프로젝트 및 JIRA 이슈 리포트로부터 추출된 commit의 인지 복잡도 메트릭 (JIT_SDP)                                                                                                                                                                                                                                                                                                       |
-| `*/README.md`    | 각 데이터셋의 통계적 요약 문서                                                                                                                                                                                                                                                                                                                                                                           |
-| **SDP/Explainer**                 |                                                                                                                                                                                                                                                                                                                                                                                                         |
-| [`SQAPlanner/LORMIKA.py`](./SDP/Explainer/SQAPlanner/LORMIKA.py) | SQAPlanner의 핵심 클래스로, 학습 데이터와 유사한 인스턴스를 생성하여 글로벌 모델의 예측 결과와 함께 출력함                                                                                                                                                                                                                                                                                               |
-| [`SQAPlanner/bigml_mining.py`](./SDP/Explainer/SQAPlanner/bigml_mining.py) | LORMIKA로부터 얻은 데이터 간 상관관계 분석을 위해 BigML API를 자동화 한 스크립트                                                                                                                                                                                                                                                                                                                         |
-| [`SQAPlanner/mining_sqa_rules.py`](./SDP/Explainer/SQAPlanner/mining_sqa_rules.py) | BigML 상관관계 분석 결과를 이용해 결함 유발 코드에 대한 실행가능한 지침을 생성                                                                                                                                                                                                                                                                                                                           |
-| [`LIME_HPO.py`](./SDP/Explainer/LIME_HPO.py) | Differential Evolution 알고리즘을 이용한 하이퍼파라미터 최적화 코드로 LIME의 생성 인스턴스 수를 최적화 한 LIME_HPO의 Python 구현체                                                                                                                                                                                                                                                                       |
-| [`TimeLIME.py`](./SDP/Explainer/TimeLIME.py) | 기존 구현체의 data leakage 및 논문과 다른 구현 부분을 수정한 TimeLIME 구현체                                                                                                                                                                                                                                                                                                                           |
-| **SDP**                           |                                                                                                                                                                                                                                                                                                                                                                                                         |
-| [`DeFlip.py`](./SDP/DeFlip.py)     | Counterfactuals 기반 결함 유발 코드 예측 프로토타입 모델인 DeFlip 구현체                                                                                                                                                                                                                                                                                                                                |
-| [`data_utils.py`](./SDP/data_utils.py) | 데이터 입출력, 모델 저장 관련 기능                                                                                                                                                                                                                                                                                                                                                                       |
-| [`preprocess.py`](./SDP/preprocess.py) | 중복 행 제거, AutoSpearman을 통한 상관관계 높은 특성 제거를 포함한 다양한 데이터 전처리 기능                                                                                                                                                                                                                                                                                                             |
-| [`run_explainer.py`](./SDP/run_explainer.py) | 각 Explainer들을 실행하여 예측 결과에 대한 지침을 생성하는 스크립트                                                                                                                                                                                                                                                                                                                                   |
-| [`flip_exp.py`](./SDP/flip_exp.py)   | 각 Explainer들이 제공하는 지침을 바탕으로 결함 데이터의 예측 결과가 뒤집힐 때 까지(flip) 섭동을 가하며 지침의 실행가능성을 평가하는 스크립트                                                                                                                                                                                                                                                           |
-| [`evaluate.py`](./SDP/evaluate.py)   | 각 Explainer들이 제공하는 지침의 실행가능성 및 모델의 성능을 평가하기 위한 메트릭 및 기능을 제공하는 스크립트                                                                                                                                                                                                                                                                                             |
-| **JIT_SDP**   |                                                                                                                                                                                                                                                                                                                                                                                                         |
-| [`extractor.py`](./JIT_SDP/extractor.py)  | 메소드 수준의 granularity, 로컬 캐싱을 지원하는 commit 데이터 추출 및 수집기                                                                                                                                                                                                                                                                                                                             |
-| [`cfg.py`](./JIT_SDP/cfg.py)        | Java 코드에 대한 CFG, dataflow 분석 및 UseDefGraph 구현체                                                                                                                                                                                                                                                                                                                                                |
-| [`checkstyler.py`](./JIT_SDP/checkstyler.py) | Checkstyle을 실행하고 incorrect indentations 정보를 파싱하여 RII CogC 메트릭을 계산 |
-| [`metrics.py`](./JIT_SDP/metrics.py) | 12종의 CogC 메트릭 구현 모음 |
-| [`compute_metrics.py`](./JIT_SDP/compute_metrics.py)	| commit 데이터셋으로부터 kamei, CogC 메트릭을 계산하는 스크립트 |
-|[`training.py`](./JIT_SDP/training.py) | 다양한 프로젝트 별 메트릭 구성, 전처리, ML 모델 설정 및 실행가능성 분석을 지원하는 학습 파이프라인|
-| [`analysis.py`](./JIT_SDP/analysis.py) | 데이터셋에 대한 logistic regression, ranksums, 스피어맨 상관계수, Cliff's Delta, pointbiserial 등의 상관계수 분석 |
+`(202300000001212) 결함 유발 코드 예측의 근거를 설명하는 딥러닝 기반 디버깅 워크벤치 개발` 연구 과제 산출물을 **폴더 전체** 기준으로 요약합니다.
 
+## 리포지토리 개요
+| 폴더 | 주요 내용 |
+| --- | --- |
+| `Datasets/` | SW/Kamei/CogC 메트릭 데이터셋 모음(CSV·README). |
+| `JIT_SDP/` | Bug-introducing commit 수집기, CogC 계산기, JIT-SDP 학습·분석 스크립트. |
+| `SDP/` | 릴리스 기반 SDP 전처리·학습·Explainer 실행·평가 도구. |
+| `eval-actionable-guidance-main/` | “actionable guidance” 재현용 전처리·학습·설명 계획 패키지. |
+| `DeFlip/` | 반사실 기반 Actionable SDP/JIT-SDP 재현(실험 실행, 모델, 출력, 데이터). |
+| `NeuroJIT/` | 신경생리학·이해가능성(CUF) 기반 커밋 예측 파이프라인 및 배포 패키지. |
+| `undersampling-techniques-for-technical-debt-prediction/` | 불균형 데이터 언더/오버 샘플링 실험 노트북 및 결과. |
+
+## 데이터셋 및 커밋 수집
+| 경로 | 기능 | 비고 |
+| --- | --- | --- |
+| [`Datasets/sw_metrics/`](./Datasets/sw_metrics/README.md) | 9개 프로젝트 70,278개 커밋의 파일 단위 SW 메트릭(train/test CSV, mapping, 전처리 README). | 릴리스별 split·매핑 포함. |
+| [`Datasets/kamei_metrics/`](./Datasets/kamei_metrics/README.md) | 13개 프로젝트 36,021개 커밋의 Kamei 메트릭(churn·개발자 이력·시간 간격) CSV. | 프로젝트별 `*_kamei.csv`. |
+| [`Datasets/cogc_metrics/`](./Datasets/cogc_metrics/README.md) | Kamei 커밋 대상 CogC·RII 메트릭 CSV와 지표 정의·스키마. | `*_metrics(_rii).csv` 다수. |
+| [`JIT_SDP/extractor.py`](./JIT_SDP/extractor.py) | 메소드 수준 diff 분리, 병렬·로컬 캐시, JIRA 매핑, SZZ 변형을 통한 bug-introducing commit 추출. | 17개 OSS + Yatish/Kamei/ApacheJIT 적재. |
+| [`JIT_SDP/checkstyle.jar`](./JIT_SDP/checkstyle.jar) / [`JIT_SDP/indentation_config.xml`](./JIT_SDP/indentation_config.xml) | CogC·RII 계산용 Checkstyle 바이너리와 규칙. | JIT/NeuroJIT 공용. |
+
+## CogC 메트릭 & JIT-SDP 파이프라인
+| 경로 | 기능 | 비고 |
+| --- | --- | --- |
+| [`JIT_SDP/cfg.py`](./JIT_SDP/cfg.py) | Java 제어/데이터 플로우 그래프(CFG·UseDefGraph) 생성·직렬화. | CogC·CF 탐색 입력 그래프. |
+| [`JIT_SDP/checkstyler.py`](./JIT_SDP/checkstyler.py) | Checkstyle 결과 파싱 후 RII·스타일 메트릭 집계. | `indentation_config.xml` 규칙 사용. |
+| [`JIT_SDP/metrics.py`](./JIT_SDP/metrics.py) | 12종 CogC 메트릭 계산(제어/데이터/예외 흐름 복잡도 등) 및 스키마 정의. | 그래프·AST 기반. |
+| [`JIT_SDP/compute_metrics.py`](./JIT_SDP/compute_metrics.py) | Kamei·CogC·RII 계산 병렬화, CSV/Parquet 저장, 캐싱. | 다중 프로젝트 일괄 처리. |
+| [`JIT_SDP/training.py`](./JIT_SDP/training.py) | 특성 선택·전처리 후 RF·XGBoost 등 JIT-SDP 학습·중요도 분석. | 모델 저장·설명 로깅. |
+| [`JIT_SDP/analysis.py`](./JIT_SDP/analysis.py) | Logistic regression, Ranksums, Spearman, Cliff’s Delta 상관·효과 분석. | 통계 검정·시각화. |
+
+## 릴리스 기반 SDP 파이프라인
+| 경로 | 기능 | 비고 |
+| --- | --- | --- |
+| [`SDP/preprocess.py`](./SDP/preprocess.py) | 결측·중복 제거, AutoSpearman 기반 고상관 특성 제거, 스케일링·캐싱. | train/test 분할 포함. |
+| [`SDP/data_utils.py`](./SDP/data_utils.py) | 데이터 로딩·분할, 모델·스케일러 직렬화, 시드 고정, 배치 예측 유틸. | 공통 파이프라인 헬퍼. |
+| [`SDP/run_explainer.py`](./SDP/run_explainer.py) | LIME-HPO, TimeLIME, SQAPlanner 실행 및 결과 저장. | 실험 스케줄링 지원. |
+| [`SDP/flip_exp.py`](./SDP/flip_exp.py) / [`SDP/evaluate.py`](./SDP/evaluate.py) | 설명 기반 지침 실행, ROC-AUC·PR-AUC 등 성능 평가. | 실험 반복·로그. |
+| [`SDP/analysis.py`](./SDP/analysis.py) | Ranksums, Cliff’s Delta, Spearman 통계 분석·시각화. | 재현 그래프 생성. |
+| [`SDP/hyparams.py`](./SDP/hyparams.py) | 모델·Explainer 설정값과 검색 공간 관리. | 실험 설정 일원화. |
+| [`SDP/DeFlip.py`](./SDP/DeFlip.py) | 반사실 기반 DeFlip 프로토타입(반사실 탐색·특징 중요도·일관성). | SDP용 초기 구현. |
+| [`SDP/Explainer/LIME_HPO.py`](./SDP/Explainer/LIME_HPO.py) | LIME-HPO 구현 및 Differential Evolution 기반 탐색. | 샘플링·가중치 설정. |
+| [`SDP/Explainer/TimeLIME.py`](./SDP/Explainer/TimeLIME.py) | 시간 제약 반영·데이터 유출 보정 TimeLIME. | 시계열 안전성 보강. |
+| [`SDP/Explainer/SQAPlanner/LORMIKA.py`](./SDP/Explainer/SQAPlanner/LORMIKA.py) | LORMIKA 전략으로 이웃 생성·지침 생성. | SQAPlanner 핵심. |
+| [`SDP/Explainer/SQAPlanner/bigml_mining.py`](./SDP/Explainer/SQAPlanner/bigml_mining.py) | BigML API 자동화 상관 규칙 채굴. | 웹 UI 의존 제거. |
+| [`SDP/Explainer/SQAPlanner/mining_sqa_rules.py`](./SDP/Explainer/SQAPlanner/mining_sqa_rules.py) | 채굴 규칙을 실행 가능한 지침 스크립트로 변환. | 지침 후처리. |
+
+## Actionable Guidance 신뢰성 재현 패키지
+| 경로 | 기능 | 비고 |
+| --- | --- | --- |
+| [`eval-actionable-guidance-main/preprocess.py`](./eval-actionable-guidance-main/preprocess.py) | 릴리스 분할·AutoSpearman 특성 선택. | 데이터셋 준비. |
+| [`eval-actionable-guidance-main/train_models.py`](./eval-actionable-guidance-main/train_models.py) | 프로젝트별 모델 학습·저장. | 모델 선택 로깅. |
+| [`eval-actionable-guidance-main/run_explainer.py`](./eval-actionable-guidance-main/run_explainer.py) | Explainer 실행 및 결과 적재(LIME/TimeLIME/SQAPlanner). | |
+| [`eval-actionable-guidance-main/plan_explanations.py`](./eval-actionable-guidance-main/plan_explanations.py) | 제안 변경안 최소화·탐색 기반 계획(flip/closest). | |
+| [`eval-actionable-guidance-main/mining_sqa_rules.py`](./eval-actionable-guidance-main/mining_sqa_rules.py) | BigML API 상관 규칙 채굴 자동화. | 수작업 제거. |
+| [`eval-actionable-guidance-main/flip_exp.py`](./eval-actionable-guidance-main/flip_exp.py) / [`eval-actionable-guidance-main/evaluate.py`](./eval-actionable-guidance-main/evaluate.py) | 지침 실행·RQ1–RQ3 평가(실행 가능성·성능). | |
+| [`eval-actionable-guidance-main/analysis.py`](./eval-actionable-guidance-main/analysis.py) | 통계 분석·시각화(RQ별 플롯). | |
+| [`eval-actionable-guidance-main/hyparams.py`](./eval-actionable-guidance-main/hyparams.py) | 실험 하이퍼파라미터 정의(모델·설명 설정 공유). | |
+| [`eval-actionable-guidance-main/data_utils.py`](./eval-actionable-guidance-main/data_utils.py) | 데이터 로딩·분할·캐시·시드 설정 유틸. | |
+| [`eval-actionable-guidance-main/Dataset/README.md`](./eval-actionable-guidance-main/Dataset/README.md) | 재현용 릴리스 데이터 설명 및 입력 스키마. | |
+| [`eval-actionable-guidance-main/model_evaluation/*.csv`](./eval-actionable-guidance-main/model_evaluation/) | CatBoost/XGBoost 등 모델 성능 테이블. | 벤치마크 결과. |
+| [`eval-actionable-guidance-main/evaluations/*.png`](./eval-actionable-guidance-main/evaluations/) | RQ1–RQ3 그래프 및 Flip/Feasibility 결과 이미지. | 논문 그림 재현. |
+
+## DeFlip (Counterfactual Actionable SDP/JIT-SDP)
+| 범주 | 경로 | 기능 |
+| --- | --- | --- |
+| 공통 실행 | [`DeFlip/replication_cli.py`](./DeFlip/replication_cli.py) | RQ별 실험 일괄 실행 CLI. |
+| 공통 시각화 | [`DeFlip/plot_rq1.py`](./DeFlip/plot_rq1.py) / [`DeFlip/plot_rq2.py`](./DeFlip/plot_rq2.py) / [`DeFlip/plot_rq3.py`](./DeFlip/plot_rq3.py) | RQ1–RQ3 결과 플롯 생성. |
+| 공통 자산 | [`DeFlip/requirements.txt`](./DeFlip/requirements.txt), [`DeFlip/model_evaluation/*.csv`](./DeFlip/model_evaluation/), [`DeFlip/catboost_info/*`](./DeFlip/catboost_info/), [`DeFlip/outputs/`](./DeFlip/outputs/), [`DeFlip/jit_models/`](./DeFlip/jit_models/) | 의존성, 벤치마크 테이블, CatBoost 로그, 결과·모델 아카이브. |
+| SDP 데이터 | [`DeFlip/SDP/Dataset/*`](./DeFlip/SDP/Dataset/) | 릴리스 기반 실험용 데이터·요약(`dataset_summary.*`, `feature_extremes.csv`). |
+| SDP 전처리·학습 | [`DeFlip/SDP/preprocess.py`](./DeFlip/SDP/preprocess.py), [`DeFlip/SDP/train_models.py`](./DeFlip/SDP/train_models.py), [`DeFlip/SDP/run_all_explainers.py`](./DeFlip/SDP/run_all_explainers.py) | 스케일링, 모델 학습, Explainer 일괄 실행. |
+| SDP 계획·반사실 | [`DeFlip/SDP/plan_explanations.py`](./DeFlip/SDP/plan_explanations.py), [`DeFlip/SDP/generate_closest_plans.py`](./DeFlip/SDP/generate_closest_plans.py), [`DeFlip/SDP/cf.py`](./DeFlip/SDP/cf.py) | 반사실 생성·가까운 계획 탐색. |
+| SDP 실행·평가 | [`DeFlip/SDP/run_explainer.py`](./DeFlip/SDP/run_explainer.py), [`DeFlip/SDP/flip_exp.py`](./DeFlip/SDP/flip_exp.py), [`DeFlip/SDP/flip_closest.py`](./DeFlip/SDP/flip_closest.py), [`DeFlip/SDP/evaluate_cf.py`](./DeFlip/SDP/evaluate_cf.py) | 지침 실행 및 반사실 평가. |
+| SDP 설정·규칙 | [`DeFlip/SDP/hyparams.py`](./DeFlip/SDP/hyparams.py), [`DeFlip/SDP/data_utils.py`](./DeFlip/SDP/data_utils.py), [`DeFlip/SDP/mining_sqa_rules.py`](./DeFlip/SDP/mining_sqa_rules.py) | 하이퍼파라미터, 데이터 헬퍼, 규칙 채굴. |
+| SDP Explainer | [`DeFlip/SDP/Explainer/LIME_HPO.py`](./DeFlip/SDP/Explainer/LIME_HPO.py), [`DeFlip/SDP/Explainer/TimeLIME.py`](./DeFlip/SDP/Explainer/TimeLIME.py), [`DeFlip/SDP/Explainer/SQAPlanner/LORMIKA.py`](./DeFlip/SDP/Explainer/SQAPlanner/LORMIKA.py), [`DeFlip/SDP/Explainer/SQAPlanner/bigml_mining.py`](./DeFlip/SDP/Explainer/SQAPlanner/bigml_mining.py) | LIME-HPO·TimeLIME·SQAPlanner 구현. |
+| JIT 데이터·전처리 | [`DeFlip/JIT-SDP/Dataset/*`](./DeFlip/JIT-SDP/Dataset/), [`DeFlip/JIT-SDP/preprocess_jit.py`](./DeFlip/JIT-SDP/preprocess_jit.py) | JIT-SDP 입력 데이터와 전처리. |
+| JIT 학습 | [`DeFlip/JIT-SDP/train_models_jit.py`](./DeFlip/JIT-SDP/train_models_jit.py) | JIT 모델 학습 및 저장. |
+| JIT 반사실·계획 | [`DeFlip/JIT-SDP/cf.py`](./DeFlip/JIT-SDP/cf.py), [`DeFlip/JIT-SDP/plan_explanations.py`](./DeFlip/JIT-SDP/plan_explanations.py), [`DeFlip/JIT-SDP/plan_cfe.py`](./DeFlip/JIT-SDP/plan_cfe.py), [`DeFlip/JIT-SDP/plan_closest.py`](./DeFlip/JIT-SDP/plan_closest.py), [`DeFlip/JIT-SDP/plan_pyexp.py`](./DeFlip/JIT-SDP/plan_pyexp.py) | K-Lasso·PyExplainer 등 대안 계획·근접 반사실 생성. |
+| JIT 실행·평가 | [`DeFlip/JIT-SDP/run_explainer.py`](./DeFlip/JIT-SDP/run_explainer.py), [`DeFlip/JIT-SDP/run_cfexp.py`](./DeFlip/JIT-SDP/run_cfexp.py), [`DeFlip/JIT-SDP/run_pyexp.py`](./DeFlip/JIT-SDP/run_pyexp.py), [`DeFlip/JIT-SDP/flip_exp.py`](./DeFlip/JIT-SDP/flip_exp.py), [`DeFlip/JIT-SDP/flip_closest.py`](./DeFlip/JIT-SDP/flip_closest.py), [`DeFlip/JIT-SDP/evaluate_closest.py`](./DeFlip/JIT-SDP/evaluate_closest.py), [`DeFlip/JIT-SDP/evaluate_final.py`](./DeFlip/JIT-SDP/evaluate_final.py) | 반사실 실행·최종 메트릭 산출. |
+| JIT 탐색·보조 | [`DeFlip/JIT-SDP/crossover_interpolation.py`](./DeFlip/JIT-SDP/crossover_interpolation.py), [`DeFlip/JIT-SDP/counterfactual_generation.py`](./DeFlip/JIT-SDP/counterfactual_generation.py), [`DeFlip/JIT-SDP/random_perturbatio.py`](./DeFlip/JIT-SDP/random_perturbatio.py), [`DeFlip/JIT-SDP/rule_mining.py`](./DeFlip/JIT-SDP/rule_mining.py), [`DeFlip/JIT-SDP/rulefit.py`](./DeFlip/JIT-SDP/rulefit.py), [`DeFlip/JIT-SDP/K-Lasso.py`](./DeFlip/JIT-SDP/K-Lasso.py), [`DeFlip/JIT-SDP/hyparams.py`](./DeFlip/JIT-SDP/hyparams.py), [`DeFlip/JIT-SDP/data_utils.py`](./DeFlip/JIT-SDP/data_utils.py), [`DeFlip/JIT-SDP/pyexplainer_core.py`](./DeFlip/JIT-SDP/pyexplainer_core.py) | 탐색 전략, 규칙 기반 모델, 하이퍼파라미터·데이터 유틸. |
+| JIT Explainer | [`DeFlip/JIT-SDP/Explainer/LIME_HPO.py`](./DeFlip/JIT-SDP/Explainer/LIME_HPO.py) | JIT-SDP 맞춤 LIME-HPO. |
+
+## NeuroJIT (CUF·경생리학 기반 JIT-SDP)
+| 경로 | 기능 | 비고 |
+| --- | --- | --- |
+| [`NeuroJIT/pyproject.toml`](./NeuroJIT/pyproject.toml) / [`NeuroJIT/Dockerfile`](./NeuroJIT/Dockerfile) / [`NeuroJIT/docker-compose.yml`](./NeuroJIT/docker-compose.yml) | 패키징·컨테이너 설정, 배포 구성. | `requirements.lock`/`requirements-dev.lock` 종속성. |
+| [`NeuroJIT/src/neurojit/commit.py`](./NeuroJIT/src/neurojit/commit.py) | 학습·평가 대상 커밋 필터링·저장 로직. | 패키지 진입 핵심. |
+| [`NeuroJIT/scripts/jit_sdp.py`](./NeuroJIT/scripts/jit_sdp.py) | JIT-SDP 실행 스크립트. | CLI 진입점. |
+| [`NeuroJIT/scripts/data_utils.py`](./NeuroJIT/scripts/data_utils.py) | 슬라이딩 윈도우, verification latency, concept drift 대응 데이터 구성. | 시간 순서 반영. |
+| [`NeuroJIT/scripts/calculate.py`](./NeuroJIT/scripts/calculate.py) / [`NeuroJIT/scripts/environment.py`](./NeuroJIT/scripts/environment.py) | 특징 계산 환경 설정 및 실행. | 설정·경로 관리. |
+| [`NeuroJIT/indentation_config.xml`](./NeuroJIT/indentation_config.xml) | Checkstyle 규칙(Indentation 등)으로 II/CUF 계산. | RII 공유 규칙. |
+| [`NeuroJIT/scripts/pre_analysis.py`](./NeuroJIT/scripts/pre_analysis.py) / [`NeuroJIT/scripts/analysis.py`](./NeuroJIT/scripts/analysis.py) / [`NeuroJIT/scripts/correlation.py`](./NeuroJIT/scripts/correlation.py) | 메트릭 상관·효과 분석 파이프라인(Cliff’s Delta 등). | |
+| [`NeuroJIT/scripts/visualization.py`](./NeuroJIT/scripts/visualization.py) | 분석·모델 결과 시각화. | 재현 그래프. |
+| [`NeuroJIT/scripts/reproduce.sh`](./NeuroJIT/scripts/reproduce.sh) / [`NeuroJIT/scripts/extract_pickles.sh`](./NeuroJIT/scripts/extract_pickles.sh) | 결과 재현·피클 캐시 추출. | CLI·배치 실행. |
+| [`NeuroJIT/data/dataset/*.csv`](./NeuroJIT/data/dataset/) / [`NeuroJIT/data/output/*.json`](./NeuroJIT/data/output/) | baseline/ApacheJIT/Kamei 메트릭과 모델 성능 JSON. | 실험 데이터 자산. |
+| [`NeuroJIT/archive/pickles_part_*`](./NeuroJIT/archive/) | 슬라이스된 피클 캐시. | 대규모 데이터 저장. |
+| [`NeuroJIT/dist/neurojit-1.0.2-py3-none-any.whl`](./NeuroJIT/dist/neurojit-1.0.2-py3-none-any.whl) | 배포용 휠. | PyPI 호환. |
+| [`NeuroJIT/README.md`](./NeuroJIT/README.md) / [`NeuroJIT/README_PYPI.md`](./NeuroJIT/README_PYPI.md) / [`NeuroJIT/reproduced_results.png`](./NeuroJIT/reproduced_results.png) | 사용법 문서와 재현 결과 이미지. | 패키지 소개. |
+
+## 불균형 데이터 샘플링 실험
+| 경로 | 기능 | 비고 |
+| --- | --- | --- |
+| [`undersampling-techniques-for-technical-debt-prediction/under_over_sampling_scripts.ipynb`](./undersampling-techniques-for-technical-debt-prediction/under_over_sampling_scripts.ipynb) | 언더/오버 샘플링 적용 및 분류기별 성능 비교 실험 노트북. | 실행 기록 포함. |
+| [`undersampling-techniques-for-technical-debt-prediction/Results_for_all_tables.csv`](./undersampling-techniques-for-technical-debt-prediction/Results_for_all_tables.csv) | 실험 결과 테이블. | 요약 수치. |
+| [`undersampling-techniques-for-technical-debt-prediction/X.csv`](./undersampling-techniques-for-technical-debt-prediction/X.csv) / [`Y.csv`](./undersampling-techniques-for-technical-debt-prediction/Y.csv) | 입력·레이블 데이터. | 학습 데이터 원본. |
+| [`undersampling-techniques-for-technical-debt-prediction/readME.txt`](./undersampling-techniques-for-technical-debt-prediction/readME.txt) | 노트북 실행 가이드 및 데이터 설명. | |
 
 ## Acknowledgement
-이 성과는 정부(과학기술정보통신부)의 재원으로 한국연구재단의 지원을 받아 수행된 연구임(NRF-2023R1A2C1006390). 
+이 성과는 정부(과학기술정보통신부)의 재원으로 한국연구재단의 지원을 받아 수행된 연구임(NRF-2023R1A2C1006390).
 
 This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (NRF-2023R1A2C1006390).
