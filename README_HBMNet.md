# HBMNet: GPU NoC Simulator for MoE Workloads

BookSim2 기반 GPU Network-on-Chip 시뮬레이터로, HBM Fabric 토폴로지에서 Mixture-of-Experts (MoE) 트래픽을 시뮬레이션합니다.

#### Table of Contents

- [Topology](#topology)
- [Routing Functions](#routing-functions)
- [Deadlock Freedom (Duato's Protocol)](#deadlock-freedom-duatos-protocol)
- [Configuration](#configuration)
- [Running Simulations](#running-simulations)
- [Batch Runner (run\_all\_moe.py)](#batch-runner-run_all_moepy)
- [Traffic Matrices](#traffic-matrices)

---

## Topology

`hbmnet` 토폴로지는 GPU 내부 NoC를 모델링합니다.

### Router 구조


| Router     | 역할               | 연결 노드                |
| ------------ | -------------------- | -------------------------- |
| Crossbar 0 | SM 파티션 0 (상단) | SM 0 ~ SM 73             |
| Crossbar 1 | SM 파티션 1 (하단) | SM 74 ~ SM 147           |
| HBM 0 ~ 7  | HBM 메모리 스택    | L2 슬라이스 (HBM당 32개) |

### 링크 구조

- **Crossbar-HBM**: 각 HBM이 자신이 속한 파티션의 Crossbar에 연결
- **Crossbar-Crossbar**: 두 파티션 간 내부 연결

### `is_fabric` 에 따른 토폴로지 변화


| `is_fabric` | 토폴로지        | 설명                                                                                       |
| ------------- | ----------------- | -------------------------------------------------------------------------------------------- |
| **0**       | Tree (baseline) | Crossbar-HBM, Crossbar-Crossbar 링크만 존재. 트래픽은 반드시 Crossbar를 경유               |
| **1**       | Tree + Fabric   | **HBM-HBM 수직 인접 링크** 추가. 같은 컬럼 내에서 수직으로 인접한 HBM 라우터끼리 직접 연결 |

HBM 물리적 배치 (2열 x K/2행):

```
Column 0    Column 1
  HBM0        HBM4      <- partition 0
  HBM1        HBM5
  [Crossbar 0]
  [Crossbar 1]
  HBM2        HBM6      <- partition 1
  HBM3        HBM7
```

Fabric이 활성화되면 같은 열 내 수직 인접 HBM 간에 직접 링크가 추가됩니다 (예: HBM0-HBM1, HBM1-HBM2 등). GPU 다이의 물리적 구조상 좌우 열 간 수평 링크는 없습니다.

---

## Routing Functions

6가지 라우팅 함수를 지원합니다:


| Routing         | 설명                                                                                                             | 적합한 환경                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| `baseline`      | 결정적 트리 라우팅. SM->Xbar->HBM 또는 HBM->Xbar->HBM 고정 경로                                                  | `is_fabric=0` (Fabric 없음)   |
| `min_oblivious` | 최소 경로 중 균일 랜덤 선택. 혼잡도 미반영                                                                       | Fabric 환경, 간단한 부하 분산 |
| `min_adaptive`  | 적응형 최소 경로. 각 방향의 크레딧 사용량 기반으로 최적 방향 선택                                                | Fabric 환경, 적응형 라우팅    |
| `valiant`       | 비최소 경로 (Oblivious). 항상 랜덤 중간 HBM을 경유                                                               | Fabric 환경, 부하 균등 분배   |
| `ugal`          | UGAL (Universal Globally-Adaptive Load-balanced). 주입 시 최소/비최소 경로의 비용(거리 x 혼잡도)을 비교하여 선택 | Fabric 환경, 최적 적응형      |
| `hybrid`        | 확률적 혼합.`baseline_ratio` 비율로 baseline, 나머지는 `hybrid_routing`으로 지정한 함수 사용                     | 혼합 실험                     |

### Hybrid Routing 상세

Hybrid 라우팅은 플릿(flit) 단위로 주입 시점에 확률적 결정을 합니다:

1. 랜덤 값 생성 (0~9999)
2. `baseline_ratio * 10000` 미만이면 -> baseline 트리 라우팅
3. 그 이상이면 -> `hybrid_routing`으로 지정한 함수 (ugal, min_adaptive, valiant 등)

```
// Config 예시: 50% baseline, 50% ugal
routing_function = hybrid;
baseline_ratio = 0.5;
hybrid_routing = ugal;
```

`baseline_ratio`를 조절하여 baseline과 adaptive 라우팅의 비율을 자유롭게 변경할 수 있습니다.

---

## Deadlock Freedom (Duato's Protocol)

**공정한 실험을 위해, baseline을 포함한 모든 라우팅 함수에 Duato의 deadlock-free 알고리즘이 적용됩니다.**

### Virtual Channel 구조


| VC                | 용도                         | 라우팅                              |
| ------------------- | ------------------------------ | ------------------------------------- |
| VC 0 (Escape)     | Deadlock 방지용 (저우선순위) | 항상 결정적 baseline 트리 경로 사용 |
| VC 1 ~ V-1 (Data) | 일반 트래픽 (고우선순위)     | 선택한 라우팅 알고리즘 사용         |

- Data VC (1~V-1)에서 비최소/적응형 라우팅으로 인한 사이클이 발생할 수 있음
- Escape VC (0)는 비순환(acyclic) 트리 경로를 사용하여 항상 진행 보장
- Deadlock 상태의 플릿은 Escape VC로 전환하여 탈출 가능

이를 통해 baseline과 다른 라우팅 함수 간의 비교 시, deadlock freedom이라는 동일한 조건 하에서 공정한 성능 비교가 가능합니다.

---

## Configuration

주요 설정 파일: `src/examples/hbmnet_config`

### 주요 파라미터

```
// Topology
topology = hbmnet;
num_sms = 148;            // SM 개수
num_l2_slices = 256;      // L2 슬라이스 개수
num_hbm_stacks = 8;       // HBM 스택 수

// Fabric
is_fabric = 1;            // 0: tree only, 1: tree + HBM fabric

// Routing
routing_function = ugal;  // baseline/min_oblivious/min_adaptive/ugal/valiant/hybrid
ugal_threshold = 50;      // UGAL cost threshold
baseline_ratio = 0.5;     // hybrid 전용: baseline 비율
hybrid_routing = ugal;    // hybrid 전용: 나머지 라우팅 함수

// Link latencies (cycles)
xbar_xbar_latency = 185;
xbar_hbm_latency = 170;
hbm_hbm_latency = 170;

// Link bandwidths (parallel ports per link)
xbar_xbar_bandwidth = 70;
xbar_hbm_bandwidth = 14;
hbm_hbm_bandwidth = 14;

// Flow control
num_vcs = 4;              // VC 0 = escape, VC 1~3 = data
vc_buf_size = 256;

// Simulation
sim_type = moe;
traffic_matrix_file = ./examples/end-to-end/moe_matrix_H2H_k16_128MiB;
```

---

## Running Simulations

### 단일 실행

```bash
cd booksim2/src
./booksim ./examples/hbmnet_config
```

Config 파라미터를 직접 수정 or 커맨드라인으로 오버라이드할 수 있습니다:

```bash
./booksim ./examples/hbmnet_config \
  routing_function=ugal \
  is_fabric=1 \
  traffic_matrix_file=./examples/end-to-end/moe_matrix_baseline_k1_8MiB
```

---

## Batch Runner (run_all_moe.py)

MoE 트래픽 시뮬레이션을 일괄 실행하고 결과를 CSV로 정리하는 스크립트입니다.

### 기본 사용법

```bash
cd booksim2
python3 run_all_moe.py
```

실행 전 계획된 모든 작업을 테이블로 표시하고, Enter를 누르면 실행을 시작합니다.

### 주요 옵션


| 옵션              | 설명                           | 기본값                              |
| ------------------- | -------------------------------- | ------------------------------------- |
| `--schemes`       | 실행할 시나리오 선택           | 전체 (Baseline, Fabric, Offloading) |
| `--routings`      | 라우팅 함수 지정 (복수 가능)   | 시나리오 기본값                     |
| `--k-values`      | MoE top-k 값 필터              | 전체 (1, 2, 4, 8, 16)               |
| `--sizes`         | 메시지 크기(MiB) 필터          | 전체 (8, 16, 32, 64, 128, 256)      |
| `--workers`       | 병렬 워커 수                   | CPU 코어 수                         |
| `--dry-run`       | 명령만 출력, 실행 안 함        | -                                   |
| `--parse-only`    | 기존 결과만 파싱               | -                                   |
| `--skip-existing` | 기존 결과 파일이 있으면 건너뜀 | -                                   |
| `--yes` / `-y`    | 확인 프롬프트 건너뜀           | -                                   |
| `--booksim`       | booksim 바이너리 경로          | `./src/booksim`                     |

### 사용 예시

```bash
# Fabric 시나리오에서 ugal, valiant, min_adaptive 라우팅 비교
python3 run_all_moe.py --routings ugal valiant min_adaptive --schemes Fabric

# k=1, 4만, 크기 8, 64MiB만 실행
python3 run_all_moe.py --k-values 1 4 --sizes 8 64

# 기존 결과 파싱만 수행
python3 run_all_moe.py --parse-only

# 확인 없이 바로 실행
python3 run_all_moe.py -y --skip-existing
```

### 시나리오 (Scenarios)


| 시나리오   | is_fabric | 기본 라우팅 | 트래픽 매트릭스     | 설명                           |
| ------------ | ----------- | ------------- | --------------------- | -------------------------------- |
| Baseline   | 0         | baseline    | moe_matrix_baseline | Fabric 없는 기본 트리 토폴로지 |
| Fabric     | 1         | ugal        | moe_matrix_baseline | Fabric 링크 추가, 동일 트래픽  |
| Offloading | 1         | ugal        | moe_matrix_H2H      | HBM-to-HBM 오프로딩 트래픽     |

### 출력 메트릭


| 메트릭                | 설명                        |
| ----------------------- | ----------------------------- |
| Completion Time       | MoE 배치 완료 시간 (cycles) |
| Avg Packet Latency    | 평균 패킷 지연              |
| Avg Network Latency   | 평균 네트워크 지연          |
| Avg Hops              | 평균 홉 수                  |
| UGAL Non-min %        | UGAL 비최소 경로 선택 비율  |
| Escape VC %           | Escape VC (VC 0) 사용 비율  |
| Injection/Accept Rate | 주입/수신 속도              |

### 라우팅 비교 모드

`--routings`에 복수의 라우팅 함수를 지정하면:

- 각 (시나리오, k, 크기)에 대해 모든 라우팅을 실행
- 결과 파일명에 라우팅 이름 포함 (예: `Fabric_ugal_k1_8MiB.txt`)
- 실행 완료 후 라우팅 간 성능 비교 테이블 출력

결과는 `results-test/summary.csv`에 저장되며, 작업 시작 순서대로 정렬됩니다.

---

## Traffic Matrices

### 매트릭스 형식

탭으로 구분된 텍스트 파일: 작성하신 xlsx 파일의 cell을 txt로 복사해와야 합니다.

```
Source\Dest	HBM0	HBM1	HBM2	...	HBM7		Row sum
HBM0	0.0000	27.5922	27.5922	...	3.9999		98.7765
HBM1	27.5922	0.0000	27.5922	...	3.9999		98.7765
...
Col sum	...
```

값은 MB 단위의 트래픽량을 나타냅니다.

### 지원하는 Source/Destination 레이블

MoE Traffic Manager는 다양한 granularity의 레이블을 지원합니다:


| 레이블 형식           | 의미                            | 매핑                        |
| ----------------------- | --------------------------------- | ----------------------------- |
| `HBM{h}`              | HBM 스택 h (라우터 레벨)        | 해당 HBM의 모든 L2 슬라이스 |
| `L2_{i}`              | 개별 L2 슬라이스 i              | 단일 L2 노드                |
| `Core{c}` / `Xbar{p}` | Crossbar 파티션 c (라우터 레벨) | 해당 파티션의 모든 SM       |
| `SM{i}`               | 개별 SM i                       | 단일 SM 노드                |

이를 통해 라우터 레벨(HBM, Core)부터 개별 노드 레벨(L2 슬라이스, SM)까지 다양한 granularity의 트래픽 매트릭스를 사용할 수 있습니다.

### 매트릭스 디렉토리


| 디렉토리                    | 레이블                   | 설명                            |
| ----------------------------- | -------------------------- | --------------------------------- |
| `src/examples/end-to-end/`  | HBM0~7 (8x8)             | HBM-to-HBM 전체 경로 매트릭스   |
| `src/examples/moe-GPU-HBM/` | Core0~1 + HBM0~7 (10x10) | SM->HBM + HBM<->HBM 트래픽 포함 |

### 새 매트릭스 추가 방법

1. **128MiB base 매트릭스 생성**: k 값별로 `moe_matrix_{mode}_k{k}_128MiB` 파일을 해당 디렉토리에 추가
2. **gen_all_matrices.py 실행**: 128MiB 매트릭스를 기반으로 8, 16, 32, 64, 256 MiB 크기로 자동 확장

### gen_all_matrices.py

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


| Target  | Scale Factor |
| --------- | -------------- |
| 8 MiB   | x0.0625      |
| 16 MiB  | x0.125       |
| 32 MiB  | x0.25        |
| 64 MiB  | x0.5         |
| 256 MiB | x2.0         |

#### Granularity 확장

라우터 레벨 매트릭스를 개별 노드 레벨로 확장합니다. 총 트래픽량이 보존되도록 셀 값을 `row_expansion x col_expansion`으로 나눕니다.

**end-to-end (HBM -> L2 슬라이스):**


| `--expand` | 결과             | 매트릭스 크기 |
| ------------ | ------------------ | --------------- |
| `none`     | HBM 레벨         | 8x8           |
| `l2`       | L2 슬라이스 레벨 | 256x256       |

**moe-GPU-HBM (Core/HBM -> SM/L2):**


| `--expand` | 결과                    | 매트릭스 크기 |
| ------------ | ------------------------- | --------------- |
| `none`     | Core + HBM 레벨         | 10x10         |
| `sm`       | SM + HBM 레벨           | 156x156       |
| `l2`       | Core + L2 슬라이스 레벨 | 258x258       |
| `all`      | SM + L2 슬라이스 레벨   | 404x404       |

생성되는 파일명 패턴: `moe_matrix_{mode}_k{k}_{size}MiB[_sm][_l2]`

#### gen_all_matrices.py 옵션


| 옵션          | 설명                         | 기본값        |
| --------------- | ------------------------------ | --------------- |
| `--dir`       | 매트릭스 디렉토리            | 스크립트 위치 |
| `--expand`    | 확장 granularity             | none          |
| `--force`     | 기존 파일 덮어쓰기           | -             |
| `--num-hbm`   | HBM 스택 수                  | 8             |
| `--num-l2`    | L2 슬라이스 수               | 256           |
| `--num-sms`   | SM 수 (moe-GPU-HBM만)        | 148           |
| `--num-cores` | Core 그룹 수 (moe-GPU-HBM만) | 2             |
