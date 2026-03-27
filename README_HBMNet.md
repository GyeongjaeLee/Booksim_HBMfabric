# HBMNet: GPU NoC Simulator for MoE Workloads

BookSim2 기반 GPU Network-on-Chip 시뮬레이터로, HBM Fabric 토폴로지에서 Mixture-of-Experts (MoE) 트래픽 및 AccelSim L2↔HBM 트래픽을 시뮬레이션합니다.

#### Table of Contents

- [Topology](#topology)
- [Routing Functions](#routing-functions)
- [Deadlock Freedom (Duato's Protocol)](#deadlock-freedom-duatos-protocol)
- [Configuration](#configuration)
- [Running Simulations](#running-simulations)
- [Batch Runner (run\_all\_moe.py)](#batch-runner-run_all_moepy)
- [Link Stats Visualizer (plot\_link\_stats.py)](#link-stats-visualizer-plot_link_statspy)
- [Traffic Matrices](#traffic-matrices)

---

## Topology

`hbmnet_accelsim` 토폴로지는 GPU 내부 NoC를 모델링합니다. **Xbar(Crossbar) 수 `P`와 Xbar당 HBM 스택 수 `H`를 config로 자유롭게 설정**하여 다양한 GPU 아키텍처를 표현할 수 있습니다.

### 파라미터 구성

| 파라미터 | 의미 | 관계 |
|---|---|---|
| `num_xbars` (P) | Crossbar 라우터 수 | - |
| `hbm_per_side` (H) | Xbar 한쪽에 연결된 HBM 스택 수 | - |
| `K = P × H × 2` | 총 HBM 스택 수 (양쪽 합산) | - |
| `num_sms` (N) | SM 수 (P의 배수여야 함) | SM당 N/P개씩 각 Xbar에 연결 |
| `num_l2_slices` (M) | L2 슬라이스 수 (K의 배수여야 함) | L2당 M/K개씩 각 HBM에 연결 |

### 지원 아키텍처 예시

| 구조명 | P (num\_xbars) | H (hbm\_per\_side) | K (HBM 총수) |
|---|---|---|---|
| B100\_Local | 1 | 2 | 4 |
| H100 | 1 | 3 | 6 |
| B100\_Global / B200 | 2 | 2 | 8 |
| B100\_Core\_Rotate | 2 | 3 | 12 |
| Rubin\_Ultra | 4 | 2 | 16 |

### Router 구성

| 라우터 | 수 | 역할 |
|---|---|---|
| Xbar 0 ~ P-1 | P | SM 파티션 라우터. 각 파티션에 N/P개 SM 연결 |
| MC 0 ~ K-1 | K | Memory Controller 라우터. miss path 중간 경유 |
| HBM 0 ~ K-1 | K | HBM 스택 라우터. 각 HBM에 M/K개 L2 슬라이스 연결 |

### 링크 구조

| 링크 타입 | 경로 | 설명 |
|---|---|---|
| `XBAR_HBM` | Xbar ↔ HBM | Hit path (L2 cache hit 시 직접 연결) |
| `XBAR_MC` | Xbar ↔ MC | Miss path 진입 (L2 cache miss 시) |
| `MC_HBM` | MC ↔ HBM | Miss path 로컬 (MC_h ↔ HBM_h 1:1 연결) |
| `MC_MC` | MC ↔ MC | Miss path fabric (같은 열, 인접 행 연결) |
| `XBAR_XBAR` | Xbar ↔ Xbar | Xbar 간 선형 체인 연결 |

### `is_fabric`에 따른 토폴로지 변화

| `is_fabric` | 활성 링크 | 설명 |
|---|---|---|
| **0** | XBAR\_HBM, XBAR\_MC, MC\_HBM, XBAR\_XBAR | Baseline. MC fabric 없음 |
| **1** | 위 전체 + **MC\_MC** | Fabric 활성화. 같은 열 내 인접 MC 간 직접 링크 추가 |

### 물리 배치 (B200 예시: P=2, H=2, K=8)

P개의 Xbar가 수직 방향으로 체인 연결되며, 각 Xbar의 **왼쪽(Col 0)**과 **오른쪽(Col 1)**에 각각 H개의 MC/HBM이 위치합니다. 같은 열 내 수직 인접 MC끼리만 MC\_MC 링크로 연결되며, 행 방향(Col 0 ↔ Col 1) 직접 연결은 없습니다(Xbar가 중간에 위치).

```
   Col 0              Col 1

   MC0 ─┐         ┌─ MC4
    ↕    │  Xbar0  │    ↕
   MC1 ─┘    │    └─ MC5
    ↕         │         ↕    ← crosspartition MC_MC links
   MC2 ─┐    │    ┌─ MC6
    ↕    │  Xbar1  │    ↕
   MC3 ─┘         └─ MC7
```

`↕` : MC\_MC fabric 링크 (is\_fabric=1 시 활성). 중앙 `│` : Xbar 체인 (XBAR\_XBAR) 링크. HBM은 각 MC에 1:1로 대응됩니다.

### Rubin Ultra 배치 (P=4, H=2, K=16)

```
   Col 0              Col 1

   MC0 ─┐         ┌─ MC8
    ↕    │  Xbar0  │    ↕
   MC1 ─┘    │    └─ MC9
    ↕         │         ↕    ← crosspartition
   MC2 ─┐    │    ┌─ MC10
    ↕    │  Xbar1  │    ↕
   MC3 ─┘    │    └─ MC11
    ↕         │         ↕    ← crosspartition
   MC4 ─┐    │    ┌─ MC12
    ↕    │  Xbar2  │    ↕
   MC5 ─┘    │    └─ MC13
    ↕         │         ↕    ← crosspartition
   MC6 ─┐    │    ┌─ MC14
    ↕    │  Xbar3  │    ↕
   MC7 ─┘         └─ MC15
```

Xbar 4개가 선형 체인(Xbar0─Xbar1─Xbar2─Xbar3)으로 연결됩니다. MC\_MC fabric 링크는 같은 열 내 수직 인접 MC 간에만 형성됩니다.

---

## Routing Functions

7가지 라우팅 함수를 지원합니다:

| Routing | 설명 | 적합한 환경 |
|---|---|---|
| `baseline` | 결정적 트리 라우팅. Xbar → MC → HBM 고정 경로 | `is_fabric=0` (Fabric 없음) |
| `min_oblivious` | 최소 경로 중 균일 랜덤 선택. 혼잡도 미반영 | Fabric 환경, 간단한 부하 분산 |
| `min_adaptive` | 적응형 최소 경로. 각 방향의 크레딧 사용량 기반으로 최적 방향 선택 | Fabric 환경, 적응형 라우팅 |
| `near_min_adaptive` | 혼잡 시 최대 1회 근최소 경로(same-distance 라우터) 우회 허용 | Fabric 환경, 불균등 병목 완화 |
| `valiant` | 비최소 경로 (Oblivious). 항상 랜덤 중간 MC를 경유 | Fabric 환경, 부하 균등 분배 |
| `ugal` | UGAL. 주입 시 최소/비최소 경로의 비용(거리 × 혼잡도)을 비교하여 선택 | Fabric 환경, 최적 적응형 |
| `hybrid` | `baseline_ratio` 비율로 baseline, 나머지는 `hybrid_routing`으로 지정한 함수 사용 | 혼합 실험 |

### min\_adaptive의 한계

`min_adaptive`는 각 방향의 크레딧(버퍼 점유)을 기준으로 최소 경로 중 가장 덜 혼잡한 방향을 선택합니다. 그러나 다음과 같은 구조적 한계가 있습니다:

- **MC\_MC fabric 링크 활용 저조**: moe-GPU-HBM 트래픽 및 AccelSim SM→L2, L2→SM 트래픽에서, Fabric scheme에서 `min_adaptive` 사용 시 MC\_MC 링크 중 **crosspartition 경계에 위치한 MC1↔MC2, MC5↔MC6 2쌍만 사용**됨. 나머지 MC↔MC 링크는 사용되지 않음
- **경로 다양성 부족**: minimal path가 항상 crosspartition 경계의 MC↔Xbar를 경유하므로, 해당 링크에 트래픽이 집중되어 **hot spot**을 형성함
- **Rubin Ultra에서 심화**: Xbar 체인이 길어지면(P=4) 연속된 Xbar↔Xbar 링크가 minimal path의 대부분을 차지하게 되어, MC fabric 링크 사용 빈도가 급격히 감소함. 경로가 길수록 MC\_MC 대신 Xbar\_Xbar 체인을 타는 것이 항상 minimum이 되기 때문

### Near-Min Adaptive Routing

`near_min_adaptive`는 `min_adaptive`의 경로 다양성 부족을 보완하여, congestion 기반으로 **최대 1회 non-minimum 경로 우회**를 허용하는 라우팅입니다.

#### 동작 원리

1. 각 라우팅 결정 시점에서 **최소 경로(min)** 방향과 **근최소 경로(near-min)** 방향을 모두 후보로 수집
   - **min 방향**: 목적지까지 거리가 1 감소하는 인접 라우터 방향
   - **near-min 방향**: 목적지까지 거리가 현재와 **동일한(same-distance)** 인접 라우터 방향. 즉, minimum이 아닌 경로로 1홉 이동
2. **선택 조건**: near-minimum 경로 link의 queue 대기시간(saturation) + latency가 minimum 경로의 congestion(credit 사용량)보다 유리할 때 near-min 선택
3. **최대 1회 우회**: 패킷 당 `nm_budget=1`로 제한. 한 번 near-min 경로를 선택하면 이후에는 min 경로만 사용

#### Cost 함수

```
min cost      = saturation + (1 - p) × (latency / max_latency) × (1 + saturation)
near-min cost = saturation + p × (latency / max_latency) × (1 + saturation)
```

- `saturation`: 해당 방향 link의 평균 credit 사용률 (buffer 점유 비율)
- `latency`: 해당 link의 전파 지연
- `(1 + saturation)` factor: 이미 혼잡한 우회 경로에 추가 트래픽이 쌓이는 것을 방지

#### `near_min_penalty` 파라미터 (p)

`p`는 near-min 경로의 latency penalty를 조절합니다:

| p 값 | 동작 |
|---|---|
| `0.0` | latency penalty 없음. **순수 congestion만 비교** → near-min을 공격적으로 선택 |
| `1.0` (기본값) | near-min에 full latency penalty 부과. 보수적 동작, min 경로 우선 |
| `> 1.0` | near-min을 더 강하게 억제. `min_adaptive`에 근접한 동작 |

```
// Config 예시
near_min_k = 1;           // 허용 우회 횟수 (현재 1 고정)
near_min_penalty = 0.5;   // 0.0=공격적, 1.0=보수적
```

#### 기대 효과

**Path diversity에 의한 성능 향상**이 핵심입니다. 불균등한 병목 지점에서 최대 1회 우회를 허용하여 트래픽을 분산시킵니다.

- **Crosspartition 경계 hot spot 완화**: 두 crosspartition이 인접한 부분의 MC↔Xbar 링크는 minimum path에서도 반드시 경유하기 때문에 hot spot이 되기 쉬움. Near-min은 **MC→MC fabric 링크를 통해 crosspartition에서 먼 MC↔Xbar 사용을 유도**하여 hot spot을 분산
- **AccelSim workload에서 효과 예상**: AccelSim SM↔L2 트래픽은 시간에 따라 점진적으로 병목이 형성되므로, 병목 발생 시 near-min 경로로 우회하여 congestion 완화 효과가 있음
- **Rubin Ultra에서 Xbar 우회 경로 제공**: Xbar 4개 체인 구조에서 Xbar↔Xbar만으로는 경로 다양성이 부족. Near-min은 MC fabric 링크를 더 넓게 활용하여 Xbar 우회 경로를 제공

#### 효과가 제한적인 경우

- **moe-GPU-HBM 트래픽** (동시 burst): 한번에 트래픽이 주어져 동시에 모든 경로가 병목되는 상황에서는 near-min 우회를 하더라도 대안 경로 역시 포화 상태이므로 효과 없음
- `p=0.0`으로 설정 시 균일(uniform) 트래픽에서 over-diversion 발생 가능

### Hybrid Routing 상세

Hybrid 라우팅은 플릿(flit) 단위로 주입 시점에 확률적 결정을 합니다:

1. 랜덤 값 생성 (0~9999)
2. `baseline_ratio × 10000` 미만이면 → baseline 트리 라우팅 (hit path)
3. 그 이상이면 → `hybrid_routing`으로 지정한 함수 (near_min_adaptive, min_adaptive 등)

```
// Config 예시: 20% baseline (hit), 80% near_min_adaptive (miss)
routing_function = hybrid;
baseline_ratio = 0.2;
hybrid_routing = near_min_adaptive;
near_min_penalty = 0.5;
```

`baseline_ratio`를 조절하여 L2 cache hit rate를 모델링할 수 있습니다.

---

## Deadlock Freedom (Duato's Protocol)

**공정한 실험을 위해, baseline을 포함한 모든 라우팅 함수에 Duato의 deadlock-free 알고리즘이 적용됩니다.**

### Virtual Channel 구조

| VC | 용도 | 라우팅 |
|---|---|---|
| VC 0 (Escape) | Deadlock 방지용 (저우선순위) | 항상 결정적 baseline 트리 경로 사용 |
| VC 1 ~ V-1 (Data) | 일반 트래픽 (고우선순위) | 선택한 라우팅 알고리즘 사용 |

- Data VC (1~V-1)에서 비최소/적응형 라우팅으로 인한 사이클이 발생할 수 있음
- Escape VC (0)는 비순환(acyclic) 트리 경로를 사용하여 항상 진행 보장
- Deadlock 상태의 플릿은 Escape VC로 전환하여 탈출 가능

이를 통해 라우팅 함수 간 비교 시 deadlock freedom이라는 동일한 조건 하에서 공정한 성능 비교가 가능합니다.

---

## Configuration

주요 설정 파일: `src/examples/hbmnet_accelsim_config`

### 주요 파라미터

```
// Topology
topology = hbmnet_accelsim;
num_sms = 148;              // SM 개수 (num_xbars의 배수)
num_l2_slices = 256;        // L2 슬라이스 개수 (K=num_xbars*hbm_per_side*2의 배수)
num_xbars = 2;              // P: Crossbar 수 (B200=2, Rubin_Ultra=4)
hbm_per_side = 2;           // H: Xbar 한쪽당 HBM 스택 수

// Fabric
is_fabric = 1;              // 0: Baseline (MC-MC 링크 없음), 1: Fabric 활성화

// Routing
routing_function = hybrid;  // baseline / hybrid
baseline_ratio = 0.2;       // hit rate: baseline(hit) 경로 비율
hybrid_routing = near_min_adaptive; // miss path 라우팅
near_min_k = 1;             // near-min 허용 우회 횟수
near_min_penalty = 0.5;     // 0.0=공격적, 1.0=보수적

// Link latencies (cycles)
xbar_xbar_latency = 185;
xbar_hbm_latency  = 10;     // hit path
xbar_mc_latency   = 160;    // miss path 진입
mc_hbm_latency    = 80;     // DRAM 접근
mc_mc_latency     = 160;    // fabric

// Link bandwidths (parallel ports per link pair)
xbar_xbar_bandwidth = 70;   // GPU-GPU: 10 units × 7 ports
xbar_hbm_bandwidth  = 7;    // hit path
xbar_mc_bandwidth   = 7;    // miss path
mc_hbm_bandwidth    = 7;
mc_mc_bandwidth     = 7;    // fabric

// Flow control
num_vcs     = 4;            // VC 0 = escape, VC 1~3 = data
vc_buf_size = 512;

// Simulation
sim_type = latency;
traffic = uniform;
injection_rate = 0.4;
```

---

## Running Simulations

### 단일 실행

```bash
cd booksim2
./src/booksim ./src/examples/hbmnet_accelsim_config
```

Config 파라미터를 오버라이드하여 빠르게 실험할 수 있습니다:

```bash
./src/booksim ./src/examples/hbmnet_accelsim_config \
  routing_function=hybrid \
  hybrid_routing=near_min_adaptive \
  near_min_penalty=0.5 \
  num_xbars=4 hbm_per_side=2 num_l2_slices=512
```

---

## Batch Runner (run\_all\_moe.py)

MoE 트래픽 시뮬레이션을 **구조 × 대역폭 × 시나리오 × 라우팅 × k × 크기** 조합으로 일괄 실행하고 결과를 CSV로 정리하는 스크립트입니다.

### 출력 디렉토리 구조

```
results/
└── {Structure}/
    └── {Bandwidth}/
        └── {Scheme}_{routing}/
            ├── k1_8MiB.txt
            ├── k1_16MiB.txt
            └── ...
```

예시: `results/B100_Global/B100+HBM3e/Fabric_near_min_adaptive/k16_256MiB.txt`

### 기본 사용법

```bash
cd booksim2
python3 run_all_moe.py
```

실행 전 계획된 모든 작업을 테이블로 표시하고 Enter를 누르면 실행을 시작합니다.

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--structures` | 실행할 GPU 아키텍처 | 전체 |
| `--bandwidths` | 대역폭 구성 선택 | 전체 |
| `--schemes` | 시나리오 선택 | 전체 (Baseline, Fabric, Offloading) |
| `--routings` | Fabric/Offloading의 `hybrid_routing` 지정 (복수 가능) | `near_min_adaptive` |
| `--k-values` | MoE top-k 값 필터 | 전체 (1, 2, 4, 8, 16) |
| `--sizes` | 메시지 크기(MiB) 필터 | 전체 (8, 16, 32, 64, 128, 256) |
| `--workers` | 병렬 워커 수 | CPU 코어 수 |
| `--dry-run` | 명령만 출력, 실행 안 함 | - |
| `--parse-only` | 기존 결과만 파싱 | - |
| `--skip-existing` | 기존 결과 파일이 있으면 건너뜀 | - |
| `--collect DIR` | 지정 디렉토리의 결과 파일을 CSV로 수집 | - |
| `--result-dir` | 결과 저장 경로 | `./results` |
| `--yes` / `-y` | 확인 프롬프트 건너뜀 | - |
| `--booksim` | booksim 바이너리 경로 | `./src/booksim` |

### 지원 구조 (Structures)

| 구조명 | num\_xbars | hbm\_per\_side | K |
|---|---|---|---|
| B100\_Local | 1 | 2 | 4 |
| H100 | 1 | 3 | 6 |
| B100\_Global | 2 | 2 | 8 |
| B100\_Core\_Rotate | 2 | 3 | 12 |
| Rubin\_Ultra | 4 | 2 | 16 |

### 지원 대역폭 (Bandwidths)

| 대역폭명 | GPU-GPU | GPU-HBM | HBM-HBM | 비고 |
|---|---|---|---|---|
| B100+HBM3e | 10 units | 1 unit | 1 unit | GPU 대역폭 집중 |
| B100+HBM4e | 10 units | 4 units | 4 units | HBM 대역폭 확대 |
| Shoreline\_ratio | 10 units | 3.33 units | 3.89 units | - |
| Aggressive\_Max | 5 units | 4 units | 4 units | HBM 최대화 |

1 unit = 7 ports. 예: GPU-GPU 10 units → `xbar_xbar_bandwidth = 70`

### 사용 예시

```bash
# 특정 구조 + 대역폭 실행
python3 run_all_moe.py --structures B100_Global Rubin_Ultra --bandwidths B100+HBM3e

# Fabric 시나리오에서 라우팅 비교
python3 run_all_moe.py --schemes Fabric \
  --routings min_adaptive near_min_adaptive \
  --structures B100_Global --bandwidths B100+HBM3e B100+HBM4e

# near_min_adaptive로 k=16, 256MiB만 실행
python3 run_all_moe.py --routings near_min_adaptive \
  --k-values 16 --sizes 256 --skip-existing

# 기존 결과 파싱만 수행 (재실행 없이 CSV 생성)
python3 run_all_moe.py --parse-only

# 특정 디렉토리에서 결과 수집
python3 run_all_moe.py --collect ./results/B100_Global ./results/Rubin_Ultra

# 확인 없이 즉시 실행
python3 run_all_moe.py -y --skip-existing
```

### 시나리오 (Scenarios)

| 시나리오 | is\_fabric | 라우팅 | 트래픽 매트릭스 | 설명 |
|---|---|---|---|---|
| Baseline | 0 | `baseline` (고정) | moe\_matrix\_baseline | Fabric 없는 기본 트리 토폴로지 |
| Fabric | 1 | `hybrid_routing` 지정값 | moe\_matrix\_baseline | MC fabric 링크 활성화, 동일 트래픽 |
| Offloading | 1 | `hybrid_routing` 지정값 | moe\_matrix\_H2H | HBM-to-HBM 오프로딩 트래픽 |

Baseline 시나리오는 항상 `routing_function=baseline`을 사용하며, `--routings` 옵션 무시.
Fabric/Offloading 시나리오는 `routing_function=hybrid`에서 `--routings`로 `hybrid_routing`을 설정합니다.

### 출력 메트릭 (CSV)

| 메트릭 | 설명 |
|---|---|
| `time_taken` | MoE 배치 완료 시간 (cycles) |
| `pkt_lat_avg` | 평균 패킷 지연 |
| `hops_avg` | 평균 홉 수 |
| `nearmin_nonmin_pct` | Near-min 결정 중 실제 우회 비율 |
| `nearmin_path_pct` | Near-min 우회를 1회 이상 사용한 패킷 비율 |
| `sat_xbar_xbar` | XBAR\_XBAR 링크 평균 포화도 |
| `sat_xbar_mc` | XBAR\_MC 링크 평균 포화도 |
| `sat_mc_mc` | MC\_MC 링크 평균 포화도 |
| `escape_vc_pct` | Escape VC (VC 0) 사용 비율 |

---

## Link Stats Visualizer (plot\_link\_stats.py)

링크별 사용률과 포화도를 3D 막대 그래프로 시각화하여 **병목 지점 진단 및 경로 분산 효과**를 분석하는 스크립트입니다.

### 생성 플롯

`--metric` 옵션에 따라 두 종류의 플롯을 생성합니다:

#### 1. Link Usage (%) — `--metric util`

- **X축**: Source 라우터 (Xbar0, Xbar1, MC0 ~ MCn)
- **Y축**: Destination 라우터 (역순: MCn ~ MC0, Xbar1, Xbar0)
- **Z축**: 전체 링크 통과 횟수 대비 해당 방향 사용 비율 (%)
- **색상**: 낮음(파랑) → 높음(빨강)

#### 2. Avg Saturation (%) — `--metric sat`

- 동일 축 구성
- **Z축**: 해당 방향 링크 진입 시 평균 버퍼 포화도 (%). 높을수록 대기 시간이 길었음을 의미

각 서브플롯 상단에 요약 텍스트 박스가 표시됩니다:
- **util 플롯**: `Util: X-X=xx% X-M=xx% M-M=xx%` (링크 타입별 사용률), `Cycles=xxx Hop=x.xx`
- **sat 플롯**: `Sat: X-X=xx% X-M=xx% M-M=xx%` (링크 타입별 평균 포화도), `Cycles=xxx Hop=x.xx`

### 플롯 레이아웃

```
              Fabric_min_adaptive    Fabric_near_min_adaptive
B100+HBM3e  [  3D bar chart  ]      [  3D bar chart  ]
B100+HBM4e  [  3D bar chart  ]      [  3D bar chart  ]
```

행 = 대역폭, 열 = 시나리오\_라우팅 조합. 동일한 Z축 스케일로 표시하여 직접 비교 가능.

### 무엇을 알 수 있나

**Link 사용율 비율 + Link 진입 시 saturation 강도 비교**를 통해 다음을 분석할 수 있습니다:

| 분석 목적 | 확인 방법 |
|---|---|
| **병목 링크 식별** | Avg Saturation 플롯에서 Z값이 높은 (src, dst) 쌍 → 해당 링크가 병목 |
| **MC fabric 활용도** | Link Usage 플롯의 MC→MC 막대 높이 → near\_min\_adaptive가 fabric을 더 균등하게 사용하는지 확인 |
| **경로 분산 효과** | min\_adaptive vs near\_min\_adaptive의 Usage 분포 비교 → near-min이 hot spot을 얼마나 분산시키는지 |
| **Effective BW 향상 가능성** | 특정 링크에 집중된 사용이 분산되면 bottleneck이 해소되어 실효 대역폭 증가 가능성 |
| **Crosspartition 경계 hot spot** | 두 Xbar 파티션 경계의 MC↔Xbar 사용률이 높으면 min\_adaptive의 경로 집중 현상 확인 가능 |

### 기본 사용법

```bash
cd booksim2
python3 plot_link_stats.py --structures B100_Global \
  --bandwidths B100+HBM3e --schemes Fabric \
  --routings min_adaptive near_min_adaptive \
  --k-values 16 --sizes 256
```

결과는 화면 표시(기본값) 또는 `--output` 파일로 저장됩니다.

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--structures` | 시각화할 구조 | B100\_Global |
| `--bandwidths` | 대역폭 필터 | 전체 |
| `--schemes` | 시나리오 필터 | 전체 |
| `--routings` | 라우팅 필터 | 전체 |
| `--k-values` | k 값 필터 | 16 |
| `--sizes` | 크기(MiB) 필터 | 256 |
| `--metric` | `util` / `sat` / `both` | `both` |
| `--output` | 저장 파일 경로 (.png) | 화면 표시 |
| `--result-dir` | 결과 디렉토리 | `./results` |
| `--dpi` | 출력 해상도 | 150 |

### 사용 예시

```bash
# B100_Global: min_adaptive vs near_min_adaptive 링크 사용률 비교
python3 plot_link_stats.py \
  --structures B100_Global \
  --bandwidths B100+HBM3e B100+HBM4e \
  --schemes Fabric \
  --routings min_adaptive near_min_adaptive \
  --k-values 16 --sizes 256 \
  --metric both \
  --output link_comparison.png

# Rubin_Ultra 포화도 분석
python3 plot_link_stats.py \
  --structures Rubin_Ultra \
  --bandwidths B100+HBM3e \
  --schemes Fabric Offloading \
  --routings near_min_adaptive \
  --k-values 16 --sizes 256 \
  --metric sat

# 여러 구조 비교 (k=16, 256MiB)
python3 plot_link_stats.py \
  --structures B100_Global Rubin_Ultra \
  --bandwidths B100+HBM3e \
  --schemes Fabric \
  --routings min_adaptive near_min_adaptive \
  --k-values 16 --sizes 256
```

---

## Traffic Matrices

### 매트릭스 형식

탭으로 구분된 텍스트 파일. Source × Destination 라우터/노드 간 트래픽 (MiB 단위):

```
Source\Dest    HBM0    HBM1    HBM2   ...   Row sum
HBM0           0.000  27.592  27.592  ...    98.776
HBM1          27.592   0.000  27.592  ...    98.776
...
Col sum        ...
```

### 지원하는 Source/Destination 레이블

| 레이블 형식 | 의미 | 매핑 |
|---|---|---|
| `HBM{h}` | HBM 스택 h (라우터 레벨) | 해당 HBM의 모든 L2 슬라이스 |
| `L2_{i}` | 개별 L2 슬라이스 i | 단일 L2 노드 |
| `Core{c}` / `Xbar{p}` | Crossbar 파티션 c (라우터 레벨) | 해당 파티션의 모든 SM |
| `SM{i}` | 개별 SM i | 단일 SM 노드 |

### 매트릭스 디렉토리

| 디렉토리 | 레이블 | 설명 |
|---|---|---|
| `src/examples/end-to-end/` | HBM0~7 (8×8) | HBM-to-HBM 전체 경로 매트릭스 |
| `src/examples/moe-GPU-HBM/` | Core0~1 + HBM0~7 (10×10) | SM→HBM + HBM↔HBM 트래픽 포함 |

### 새 매트릭스 추가 방법

1. **128MiB base 매트릭스 생성**: k 값별로 `moe_matrix_{mode}_k{k}_128MiB` 파일을 해당 디렉토리에 추가
2. **gen\_all\_matrices.py 실행**: 128MiB 매트릭스를 기반으로 8, 16, 32, 64, 256 MiB 크기로 자동 확장

### gen\_all\_matrices.py

128MiB base 매트릭스에서 다양한 크기 및 granularity의 매트릭스를 생성합니다.

```bash
# end-to-end: HBM 레벨 + L2 슬라이스 확장
cd src/examples/end-to-end
python3 gen_all_matrices.py --expand l2

# moe-GPU-HBM: Core/HBM 레벨 + SM/L2 확장
cd src/examples/moe-GPU-HBM
python3 gen_all_matrices.py --expand all
```

#### 크기 확장

k = 1, 2, 4, 8, 16에 대해 128MiB 매트릭스의 모든 셀 값을 `target_size / 128` 비율로 스케일링합니다.

| Target | Scale Factor |
|---|---|
| 8 MiB | ×0.0625 |
| 16 MiB | ×0.125 |
| 32 MiB | ×0.25 |
| 64 MiB | ×0.5 |
| 256 MiB | ×2.0 |

#### Granularity 확장

**moe-GPU-HBM (Core/HBM → SM/L2):**

| `--expand` | 결과 | 매트릭스 크기 |
|---|---|---|
| `none` | Core + HBM 레벨 | 10×10 |
| `sm` | SM + HBM 레벨 | 156×156 |
| `l2` | Core + L2 슬라이스 레벨 | 258×258 |
| `all` | SM + L2 슬라이스 레벨 | 404×404 |

생성되는 파일명 패턴: `moe_matrix_{mode}_k{k}_{size}MiB[_sm][_l2]`

#### gen\_all\_matrices.py 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--dir` | 매트릭스 디렉토리 | 스크립트 위치 |
| `--expand` | 확장 granularity | none |
| `--force` | 기존 파일 덮어쓰기 | - |
| `--num-hbm` | HBM 스택 수 | 8 |
| `--num-l2` | L2 슬라이스 수 | 256 |
| `--num-sms` | SM 수 (moe-GPU-HBM만) | 148 |
| `--num-cores` | Core 그룹 수 (moe-GPU-HBM만) | 2 |
