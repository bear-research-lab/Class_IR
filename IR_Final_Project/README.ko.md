# RB5 피펫팅 프로젝트

*[English](README.md)*

Rainbow Robotics RB5 협동로봇이 피펫팅 작업을 하는 MuJoCo 시뮬레이션입니다. 프로젝트는 세
파트로 이루어지며, 각 파트는 앞 파트를 바탕으로 합니다.

| 파트 | 작성할 것 | 설명서 | 테스트 |
|---|---|---|---|
| 1. 정기구학 | `student_kinematics.py`의 `forward_kinematics()` (NumPy), `student_mujoco.py`의 `mujoco_fk()` | [PART1_FK.ko.md](PART1_FK.ko.md) | `uv run part1_fk.py` |
| 2. 역기구학 | `student_kinematics.py`의 `jacobian()`, `inverse_kinematics()`, `student_mujoco.py`의 `mujoco_jacobian()` | [PART2_IK.ko.md](PART2_IK.ko.md) | `uv run part2_ik.py --task` |
| 3. 피펫팅 핸드 | `hands/<팀이름>/hand.xml` | [PART3_HAND.ko.md](PART3_HAND.ko.md) | `uv run evaluate_hand.py` |

로봇은 여러분의 기구학으로 움직입니다. 모든 동작이 여러분의 `forward_kinematics()`와
`inverse_kinematics()`를 호출하므로, 피펫팅 작업(Part 3)은 Part 1, 2가 동작해야 실행됩니다.

제출할 파일은 없습니다. 팀별로 최종 발표에서 결과를 보여 주며, 영상은 스크립트의 `--video`
옵션으로 녹화합니다. 보여 줄 내용은 각 파트 설명서에 있습니다.
각 파트가 제대로 동작할 때의 **참고 영상**: https://youtu.be/VN9uyTWR938

## 1. 설치

**uv** 하나만 설치하면 됩니다. 알맞은 Python 버전과 필요한 패키지를 uv가 모두 설치해 줍니다.

- **Windows** (PowerShell):
  ```
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS / Linux**:
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

새 터미널을 열고 이 폴더로 이동한 뒤 실행하세요.

```
uv sync
```

처음에는 1~2분 정도 걸립니다(Python 3.12, MuJoCo, NumPy, matplotlib 다운로드).

## 2. 실행

Part 1, 2 (설명서 참고):

```
uv run part1_fk.py trajectories/joints_example.txt   # 관절 각도 재생, 여러분의 FK가 핸드 위에 그려짐
uv run part1_fk.py                                   # Part 1 테스트
uv run part2_ik.py trajectories/poses_example.txt    # 여러분의 IK로 핸드를 자세들 사이로 이동
uv run part2_ik.py --task                            # Part 2 테스트 후 기본 핸드로 작업 실행
```

IK가 동작하면 Part 3:

```
uv run pipetting.py --no-error     # 기본 핸드, 홀더가 로봇이 예상한 위치에 정확히 있을 때
uv run pipetting.py --seed 3       # 기본 핸드가 피펫을 떨어뜨리는 홀더 오차
uv run pipetting.py --seed 6       # ... 피펫을 다시 꽂을 때 끼이는 홀더 오차
uv run evaluate_hand.py --seed 0   # 무작위 피펫 5개로 핸드 점수 매기기
```

`--seed`를 빼면 실행할 때마다 홀더 오차가 무작위로 바뀝니다. 오차가 작으면 기본 핸드로도
가끔 성공하지만, 오차가 크면 실패합니다. `pipetting.py`는 항상 표준 피펫(샤프트 8 mm,
벌브 18 mm)을 쓰고 `evaluate_hand.py`는 피펫 크기도 무작위로 바꾸므로, 같은 `--seed`
번호라도 두 스크립트에서는 서로 다른 실험이 됩니다.

`pipetting.py`는 3D 뷰어와, 벌브 힘·피펫 팁 높이를 실시간으로 보여 주는 그래프 창을 엽니다.
작업이 끝나면 두 창을 모두 닫아야 종료됩니다.

> **macOS:** MuJoCo 뷰어는 `mjpython`으로 실행해야 합니다:
> `uv run mjpython pipetting.py`. 뷰어를 열지 않는 명령(`--headless`, `--visualize` 없는
> `evaluate_hand.py`)은 그냥 `uv run`으로 실행하면 됩니다.

**뷰어 조작:** 왼쪽 드래그로 회전, 오른쪽 드래그로 이동, 스크롤로 확대/축소. 부품을 더블클릭하면
선택됩니다.

## 3. 핸드 작업하기

```
hands/basic/hand.xml        기본 핸드: 복사해서 쓰고, 직접 수정하지 마세요
hands/<팀이름>/hand.xml      여러분의 설계
```

`--hand`로 여러분의 핸드를 테스트합니다.

```
uv run pipetting.py --hand hands/<팀이름>/hand.xml --playback 0.5
uv run evaluate_hand.py --hand hands/<팀이름>/hand.xml --seed 0
```

## 4. 명령 모음

| 명령 | 하는 일 |
|---|---|
| `uv run part1_fk.py` | Part 1 테스트: 여러분의 FK를 시뮬레이터와 비교 |
| `uv run part1_fk.py 파일` | 관절 각도 파일 재생. `--out poses.txt`로 FK 자세 저장, `--headless`로 뷰어 없이 |
| `uv run part2_ik.py` | Part 2 테스트: 무작위 목표와 작업 목표로 IK 검사 |
| `... --task` | ... 이어서 기본 핸드와 여러분의 IK로 피펫팅 작업 실행 |
| `uv run part2_ik.py 파일` | 자세 파일 재생. `--joint-space`, `--out joints.txt`, `--headless` |
| `uv run pipetting.py` | 뷰어에서 작업을 한 번 실행 |
| `... --no-error` | 홀더를 공칭 위치에 정확히 둠 |
| `... --seed N` | 재현 가능한 홀더 오차 (같은 N = 같은 오차) |
| `... --playback 0.5` | 뷰어에서 슬로 모션 (물리는 동일) |
| `... --headless` | 창 없이 실행, 검사 결과와 힘 요약을 출력 |
| `... --headless --plot run.png --csv run.csv` | 힘 그래프와 힘 기록 저장 |
| `... --video run.mp4` | 실행 과정을 영상으로 녹화 (`part1_fk.py 파일`, `part2_ik.py 파일`, `part2_ik.py --task`에서도 사용 가능) |
| `... --layout "Practice 2"` | `layouts.md`의 다른 배치 사용 |
| `... --graph overlay` / `none` | 그래프를 뷰어 안에 표시, 또는 표시 안 함 |
| `uv run evaluate_hand.py --seed 0` | 핸드 채점: 피펫 크기와 홀더 오차가 무작위인 실험 5회 |
| `... -n 10 --visualize 1` | 실험 10회, 첫 번째는 뷰어로 보기 |
| `... --plot forces.png` | 모든 실험의 벌브 힘 그래프 저장 |

모든 명령은 `--hand 경로/hand.xml`을 받습니다. 전체 옵션은 `--help`로 확인하세요.

## 5. 파일 구성

| 경로 | 내용 |
|---|---|
| `PART1_FK.ko.md`, `PART2_IK.ko.md`, `PART3_HAND.ko.md` | 세 파트의 설명: 할 일, 규칙, 채점, 최종 발표에서 보여 줄 것 |
| `student_kinematics.py` | NumPy로 직접 유도한 FK, 자코비안, IK (Part 1, 2). 로봇은 이것으로 움직입니다 |
| `student_mujoco.py` | 같은 FK와 자코비안을 MuJoCo 함수로 구현 (Part 1, 2) |
| `trajectories/` | 관절 각도 파일과 자세 파일 예시 |
| `hands/basic/` | 시작점이 되는 기본 핸드 (Part 3) |
| `layouts.md` | 물체 위치 (피펫 홀더, 물통, 분주 용기) |
| `part1_fk.py`, `part2_ik.py` | Part 1, 2 실행과 테스트 |
| `pipetting.py`, `evaluate_hand.py` | 작업 실행과 핸드 채점 (Part 3) |
| `rb5_sim/` | 시뮬레이터 코드: 장면, 로봇 제어기, 작업, 힘 측정, 보상. 읽어 보되 수정하지 마세요 |
| `assets/` | RB5 로봇 모델과 작업 장면 |

`layouts.md`에는 위치가 Markdown 표로 들어 있습니다(로봇 베이스 기준 x, y, 단위 mm).
최종 발표에서는 같은 형식으로 새 위치가 주어집니다:
`uv run evaluate_hand.py --layouts final.md`.

## 6. 문제 해결

- **`uv`를 찾을 수 없음:** 설치 후 새 터미널을 여세요.
- **창이 뜨지 않음** (원격 데스크톱, WSL, SSH): `--headless`를 붙이세요.
- **그래프 창이 열리지 않음:** `--graph overlay`로 뷰어 안에 그래프를 그리세요.
- **`... is missing: site 'grasp' ...`:** 핸드 파일에 필수 인터페이스가 빠졌습니다.
  `PART3_HAND.ko.md` 4장을 보세요.
- **`... implement inverse_kinematics() ...`:** 작업에는 여러분의 IK가 필요합니다. Part 1, 2를 먼저 끝내세요.

## 출처

- 로봇 모델: Rainbow Robotics RB5-850e.
  [RainbowRobotics/rbpodo_ros2](https://github.com/RainbowRobotics/rbpodo_ros2)의 URDF와
  메시를 변환했습니다(Apache License 2.0, `assets/rb5_850e/LICENSE` 참고).
- 물리 엔진: [MuJoCo](https://mujoco.org).
