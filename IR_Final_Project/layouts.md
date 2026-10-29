# Pipetting task layouts

Positions are the object centers on the table, in the RB5 base frame
(x forward, y to the robot's left), in millimeters. The robot only ever sees
these nominal positions; the simulator adds the holder error (±5 mm, ±2°) on top.

You can add your own layouts in this format and run them with
`uv run evaluate_hand.py --layouts my_layouts.md`.

## Practice 1

| object         | x (mm) | y (mm) |
|----------------|-------:|-------:|
| pipette holder |    500 |   -250 |
| water box      |    550 |      0 |
| dispense box   |    500 |    250 |

## Practice 2

| object         | x (mm) | y (mm) |
|----------------|-------:|-------:|
| pipette holder |    450 |   -200 |
| water box      |    600 |     50 |
| dispense box   |    450 |    300 |

## Practice 3

| object         | x (mm) | y (mm) |
|----------------|-------:|-------:|
| pipette holder |    550 |   -300 |
| water box      |    500 |    -50 |
| dispense box   |    550 |    200 |
