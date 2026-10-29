# Part 2: 역기구학 (Inverse Kinematics)

*[English](PART2_IK.md)*

피펫팅 프로그램은 로봇에게 관절 각도가 아니라 핸드가 있어야 할 자세를 줍니다. 역기구학(IK)은
핸드의 `grasp` 사이트 자세를 6개의 관절 각도로 바꿉니다. 이 파트에서는 Part 1의 FK를 이용해
IK를 작성하고, 마지막으로 기본 핸드로 피펫팅 작업을 실행해 봅니다.

## 1. 구현할 것

**`student_kinematics.py`**의 `inverse_kinematics()`를 채우세요.

```python
def inverse_kinematics(target, q_init, tool=np.eye(4), max_iters=200, tol=1e-6) -> (q, err)
```

| 인자 | 의미 |
|---|---|
| `target` | (4, 4) 베이스 좌표계에서 본 툴 좌표계의 목표 자세 |
| `q_init` | (6,) 시작 관절 각도 [rad]. 여기서 시작해서 같은 IK 해(branch)에 머무르세요 |
| `tool` | (4, 4) 플랜지 좌표계에서 본 툴 좌표계 (Part 1과 같음) |
| `max_iters`, `tol` | `max_iters`번 반복하거나 오차의 노름이 `tol`보다 작아지면 멈춤 |
| 반환값 `q` | (6,) `[Q_LO, Q_HI]` 범위 안의 관절 각도 |
| 반환값 `err` | `q`에서의 6차원 자세 오차의 노름 |

파일에 `pose_error(target, current)`가 주어져 있습니다. 현재 자세에서 목표 자세까지의 6차원
오차 [Δp; 회전 벡터]를 돌려줍니다. 나머지는 직접 작성하세요. 툴 좌표계의 6×6 자코비안을 구하는
함수도 포함됩니다 (FK의 관절 축과 위치로 구하는 기하학적 자코비안, 또는 유한 차분으로 구하는
수치 자코비안). 감쇠 최소제곱법(damped least squares)으로 시작하면 좋습니다.

    Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e

오차가 충분히 작아질 때까지 반복합니다. 한 번에 움직이는 크기를 제한하고 `q`가 관절 한계를
벗어나지 않게 하세요. Part 1과 마찬가지로 NumPy만 사용합니다.

## 2. 자세 파일

자세 파일은 시간에 따른 grasp 점의 위치와 방향입니다.

```
T:0  x:400 y:0    z:350
T:2  x:450 y:-100 z:350
T:4  x:450 y:-100 z:300 alpha:30 beta:90 gamma:180
```

- `T`: 시각 [s] (Part 1과 같음).
- `x`, `y`, `z`: 로봇 베이스 좌표계에서 본 grasp 점의 위치, 단위는 **mm**.
- `alpha`, `beta`, `gamma` (생략 가능): ZYZ 오일러 각으로 나타낸 방향, 단위는 **도**,
  R = Rz(α) Ry(β) Rz(γ). 생략하면 (0, 90, 180)을 씁니다. 이것은 작업 자세로, 핸드가 앞(+x)을
  향하고 피펫 축이 수직입니다.

예시는 `trajectories/poses_example.txt`에 있습니다. `part1_fk.py --out`은 이 형식으로 저장하고
`part2_ik.py --out`은 관절 각도 파일로 저장하므로, 두 파트의 결과를 서로 대조해 볼 수 있습니다.

## 3. 테스트

```
uv run part2_ik.py                                          # 채점용 테스트
uv run part2_ik.py --task                                   # ... 이어서 피펫팅 작업 실행 (뷰어)
uv run part2_ik.py --task --headless                        # ... 뷰어 없이
uv run part2_ik.py trajectories/poses_example.txt           # 뷰어에서 자세 파일 재생
uv run part2_ik.py trajectories/poses_example.txt --joint-space
uv run part2_ik.py trajectories/poses_example.txt --out my_joints.txt
```

- **테스트 1 (무작위):** 홈 자세 근처의 무작위 관절 각도로 만든 목표 50개. 모든 풀이는 홈
  자세에서 시작합니다.
- **테스트 2 (작업):** `layouts.md`의 모든 배치에 대한 피펫팅 작업의 grasp 사이트 경유점. 순서대로
  풀고, 각각 이전 해에서 시작합니다.
- 6차원 오차가 1e-4 (0.1 mm / 0.006°) 미만이고 모든 관절이 한계 안에 있으면 통과입니다. 오차는
  여러분의 FK나 반환한 `err`가 아니라 시뮬레이터로 측정합니다.
- **`--task`:** **기본 핸드**로, 홀더 오차 없이 피펫팅 작업 전체를 실행합니다. 로봇의 모든 동작이
  여러분의 IK를 씁니다. 직선 이동은 시뮬레이션 스텝마다(2 ms마다) IK를 **10번 반복만으로**
  호출하므로, 결국 수렴하는 것만으로는 부족하고 가까운 시작점에서 빠르게 수렴해야 합니다.
  IK가 맞으면 기본 핸드는 여기서 모든 항목을 통과합니다.
- **파일 재생:** 핸드가 자세 사이를 직선으로 이동합니다 (`--joint-space`를 주면 경유점마다 IK를
  한 번 풀고 관절 공간에서 보간합니다). 뷰어에 목표 자세(짧은 축)와 여러분의 FK 좌표계(긴 축)가
  표시되고, 경유점마다 관절 각도와 도달 오차가 출력됩니다.

## 4. 제출물

- `student_kinematics.py`
- `uv run part2_ik.py --task --headless`의 출력
- 경유점이 5개 이상인 직접 만든 자세 파일, 그리고 그 파일을 재생한 출력
- 보고서: 자코비안과 갱신 규칙, λ·스텝 제한·허용 오차의 값, 그리고 이 값들을 바꾸면 어떻게 되는지
  (예: 특이 자세 근처, 반복 횟수가 너무 적을 때)
