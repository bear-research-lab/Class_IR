# Part 2: 역기구학 (Inverse Kinematics)

*[English](PART2_IK.md)*

피펫팅 프로그램은 로봇에게 관절 각도가 아니라 핸드가 있어야 할 자세를 줍니다. 역기구학(IK)은
핸드의 `grasp` 사이트 자세를 6개의 관절 각도로 바꿉니다. 이 파트에서는 자코비안을 직접 유도하고
MuJoCo로도 구한 뒤, Part 1의 FK를 이용해 IK를 작성하고, 마지막으로 기본 핸드로 피펫팅 작업을
실행해 봅니다.

## 1. 구현할 것

### 1.1 직접 유도한 자코비안

**`student_kinematics.py`**의 `jacobian()`을 채우세요.

```python
def jacobian(q, tool=np.eye(4)) -> J   # 툴 좌표계의 6x6 기하학적 자코비안
```

0-2행은 툴 좌표계 원점의 선속도, 3-5행은 각속도이며 둘 다 베이스 좌표계 기준입니다. i번째 열은
i번째 관절에 해당합니다. FK의 관절 축과 관절 위치로 유도하거나(기하학적 자코비안),
`forward_kinematics()`의 유한 차분으로 계산하세요(각속도 행은 방향 변화의 `rotation_vector()`를
씁니다). NumPy만 사용합니다.

### 1.2 MuJoCo로 구하는 자코비안

**`student_mujoco.py`**의 `mujoco_jacobian(model, data, q, site="grasp")`를 채우세요. 같은 6×6
자코비안을 `mujoco.mj_jacSite`로 구합니다. 그 전에 `mj_kinematics`와 `mj_comPos`를 차례로 실행하고,
전체 자코비안에서 팔 관절 6개의 열(`model.joint(name).dofadr`)만 남기세요. 테스트가 두 자코비안을
모두 시뮬레이터와 비교하므로, 한쪽으로 다른 쪽의 실수를 찾을 수 있습니다.

### 1.3 역기구학

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
오차 [Δp; 회전 벡터]를 돌려줍니다. 이것을 1.1의 `jacobian()`과 함께 쓰세요 (로봇은 MuJoCo가
아니라 여러분의 NumPy 코드로 움직입니다). 감쇠 최소제곱법(damped least squares)으로 시작하면 좋습니다.

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

- **자코비안:** 무작위 관절 각도 20개에서 `jacobian()`과 `mujoco_jacobian()`을 시뮬레이터의
  grasp 사이트 자코비안과 비교합니다. 모든 원소의 오차가 1e-4 이내이면 통과입니다. 실패하면
  관절별 열의 최대 오차가 출력되므로 어느 관절이 틀렸는지 알 수 있습니다.
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
  속도도 중요합니다. 작업 한 번에 IK가 수만 번 호출됩니다. 유한차분 Jacobian은 반복마다 FK를
  6번 이상 호출하므로, 기하학적 Jacobian보다 실행이 몇 배 느려질 수 있습니다.
- **파일 재생:** 핸드가 자세 사이를 직선으로 이동합니다 (`--joint-space`를 주면 경유점마다 IK를
  한 번 풀고 관절 공간에서 보간합니다). 뷰어에 목표 자세(짧은 축)와 여러분의 FK 좌표계(긴 축)가
  표시되고, 경유점마다 관절 각도와 도달 오차가 출력됩니다.

## 4. 최종 발표

제출할 파일은 없습니다. 팀별로 발표하며, Part 2에서는 다음을 보여 주세요.

- 경유점이 5개 이상이고 방향 변화가 포함된, 직접 만든 자세 파일의 **영상**:
  `uv run part2_ik.py my_poses.txt --headless --video part2.mp4`로 녹화합니다.
  참고: [이렇게 보여야 합니다](https://youtu.be/VN9uyTWR938?t=10) (참고 영상은 예시 파일을 재생한
  것이니, 여러분이 만든 파일로 녹화하세요).
- 기본 핸드와 여러분의 IK로 피펫팅 작업을 수행하는 **영상**:
  `uv run part2_ik.py --task --headless --video task.mp4`로 녹화합니다.
- 채점용 테스트 결과 (`uv run part2_ik.py`). 두 자코비안 모두 포함.
- 자코비안을 어떻게 유도했는지, 그리고 MuJoCo로는 어떻게 구했는지.
- 자코비안과 갱신 규칙, λ·스텝 제한·허용 오차의 값, 그리고 이 값들을 바꾸면 어떻게 되는지
  (예: 특이 자세 근처, 반복 횟수가 너무 적을 때).
