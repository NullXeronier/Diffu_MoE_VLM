# WandB Integration Guide

이 문서는 MC-Planner에 통합된 Weights & Biases (WandB) 메트릭 모니터링 기능의 사용법을 설명합니다.

## 설정 방법

### 1. WandB 계정 설정

1. [wandb.ai](https://wandb.ai)에서 계정을 생성합니다.
2. API 키를 획득합니다: https://wandb.ai/settings
3. 터미널에서 로그인합니다:
   ```bash
   wandb login
   ```

### 2. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 설정을 수정합니다:

```bash
cp .env.example .env
# .env 파일을 편집하여 API 키와 프로젝트 설정을 입력
```

### 3. 설정 파일 수정

`configs/wandb.yaml` 파일에서 WandB 설정을 수정할 수 있습니다:

```yaml
wandb:
  enabled: true
  project: "your-project-name"
  experiment_name: "experiment-v1"
  tags: ["minecraft", "vlm", "planning"]
  log_images: true
  log_frequency: 10
```

## 사용 방법

### 기본 실행

```bash
python main.py
```

### 단일 태스크 실행

```bash
python main.py single_task=true task_name=obtain_wooden_pickaxe
```

### WandB 없이 실행

```bash
python main.py wandb.enabled=false
```

### 오프라인 모드로 실행

```bash
WANDB_MODE=offline python main.py
```

## 로깅되는 메트릭

### 스텝별 메트릭
- `episode/step`: 현재 스텝 번호
- `episode/reward`: 스텝별 리워드
- `episode/cumulative_reward`: 누적 리워드
- `episode/inventory_size`: 인벤토리 아이템 수
- `action/*`: 각 액션의 값
- `inventory/*`: 각 아이템의 개수
- `location/*`: 플레이어 위치 정보

### 에피소드별 메트릭
- `episode/duration_seconds`: 에피소드 지속 시간
- `episode/total_steps`: 총 스텝 수
- `episode/total_reward`: 총 리워드
- `episode/success`: 성공 여부
- `episode/exploration_area`: 탐험 영역
- `episode/planning_time`: 계획 수립 시간

### 태스크별 메트릭
- `task/success`: 태스크 성공 여부
- `task/completion_time`: 완료 시간
- `task/efficiency_score`: 효율성 점수
- `task/planning_iterations`: 계획 반복 횟수

### 벤치마크 메트릭
- `benchmark/overall_success_rate`: 전체 성공률
- `benchmark/average_completion_time`: 평균 완료 시간
- `benchmark/programmatic_success_rate`: 프로그래매틱 태스크 성공률
- `benchmark/creative_success_rate`: 크리에이티브 태스크 성공률

## 시각화 기능

### 이미지 로깅
환경의 RGB 관찰을 주기적으로 로깅합니다:
```yaml
wandb:
  log_images: true
  log_frequency: 10  # 10스텝마다 이미지 로깅
```

### 히스토그램
리워드, 스텝 수 등의 분포를 히스토그램으로 시각화합니다.

### 테이블
각 태스크의 상세 결과를 테이블로 정리합니다.

## 테스트

WandB 통합이 올바르게 작동하는지 테스트:

```bash
python test_wandb_integration.py
```

## 문제 해결

### 1. 로그인 문제
```bash
wandb login --relogin
```

### 2. 네트워크 문제로 인한 오프라인 실행
```bash
WANDB_MODE=offline python main.py
```

### 3. WandB 완전 비활성화
```bash
WANDB_DISABLED=true python main.py
```

### 4. 권한 문제
```bash
wandb login --relogin
# 또는
export WANDB_API_KEY=your_api_key
```

## 예제 출력

WandB 대시보드에서 다음과 같은 정보를 확인할 수 있습니다:

1. **실시간 메트릭**: 학습 진행상황 실시간 모니터링
2. **비교 분석**: 여러 실험 간 성능 비교
3. **하이퍼파라미터 추적**: 설정값과 성능의 상관관계
4. **미디어 로깅**: 게임 스크린샷과 영상
5. **시스템 모니터링**: GPU/CPU 사용률, 메모리 사용량

## 고급 설정

### 커스텀 메트릭 추가

`wandb_integration.py`의 `WandBLogger` 클래스를 수정하여 추가 메트릭을 로깅할 수 있습니다:

```python
def log_custom_metrics(self, custom_data):
    if not self.enabled:
        return
    
    wandb.log({
        "custom/my_metric": custom_data["my_metric"],
        "custom/another_metric": custom_data["another_metric"]
    })
```

### 태그와 메타데이터

실험을 더 잘 구성하기 위해 태그와 메타데이터를 활용합니다:

```yaml
wandb:
  tags: ["experiment-v2", "new-planner", "optimized"]
  notes: "Testing new planning algorithm with improved efficiency"
```
