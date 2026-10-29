"""Record a run to an MP4 file (the --video option of the scripts).

Used as a Sim step callback; renders off screen from a fixed camera, so every team's
video looks the same:

    with VideoRecorder(model, "run.mp4", overlays=[markers.draw]) as video:
        sim.callbacks.append(video)
        ...                      # run the robot
        video.stop()             # optional: stop before the end of the with-block
"""

from pathlib import Path

import imageio
import mujoco

# Camera in front of the robot, to its right, so the arm, the hand, the holder, the
# water box and the dispense box are all in view.
CAMERA = dict(lookat=(0.3, 0.0, 0.35), distance=1.6, azimuth=135.0, elevation=-25.0)


class VideoRecorder:
    def __init__(self, model: mujoco.MjModel, path: str | Path, fps: int = 30, width: int = 1280,
                 height: int = 720, overlays=()):
        """overlays: functions f(scene) that add extra geoms (e.g. FrameMarkers.draw) to each frame."""
        self.path = Path(path)
        self.fps = fps
        self.overlays = list(overlays)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = CAMERA["lookat"]
        self.camera.distance = CAMERA["distance"]
        self.camera.azimuth = CAMERA["azimuth"]
        self.camera.elevation = CAMERA["elevation"]
        self.writer = imageio.get_writer(self.path, fps=fps, quality=8)
        self.frames = 0
        self._next_time = None

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.writer is None:
            return
        if self._next_time is None:
            self._next_time = data.time
        if data.time + 1e-9 < self._next_time:
            return
        self._next_time += 1.0 / self.fps
        self.renderer.update_scene(data, self.camera)
        for draw in self.overlays:
            draw(self.renderer.scene)
        self.writer.append_data(self.renderer.render())
        self.frames += 1

    def stop(self) -> None:
        """Finish the file (later steps are not recorded)."""
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        self.renderer.close()
        print(f"Video saved to {self.path} ({self.frames / self.fps:.1f} s)")

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
