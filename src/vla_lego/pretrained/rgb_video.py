"""Lossless RGB recordings with one file per episode and exact frame readback."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


class RGBVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int = 20):
        encoder = shutil.which("ffmpeg")
        if encoder:
            command = [
                encoder,
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-g",
                "1",
                "-pix_fmt",
                "bgr0",
                "-threads",
                "2",
                str(path),
            ]
        else:
            python = os.environ.get("VLA_VIDEO_PYTHON")
            if not python:
                raise RuntimeError(
                    "Lossless recording requires existing ffmpeg or VLA_VIDEO_PYTHON/PyAV"
                )
            command = [
                python,
                "-m",
                "vla_lego.pretrained.rgb_video",
                "--encode",
                str(path),
                "--width",
                str(width),
                "--height",
                str(height),
                "--fps",
                str(fps),
            ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.sha256 = hashlib.sha256()
        self.frames = 0

    def write(self, rgb):
        payload = rgb.tobytes()
        assert self.process.stdin is not None
        self.process.stdin.write(payload)
        self.sha256.update(payload)
        self.frames += 1

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        if self.process.wait(timeout=120) != 0:
            raise RuntimeError("Lossless RGB encoder failed")


def read_rgb_frame(path: Path, index: int, *, fps: int = 20):
    import numpy as np

    if index < 0:
        raise ValueError("A recorded frame index must be non-negative")
    try:
        import av
    except ImportError:
        encoder = shutil.which("ffmpeg")
        if not encoder:
            raise RuntimeError("RGB video readback requires existing PyAV or ffmpeg") from None
        output = subprocess.check_output(
            [
                encoder,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vf",
                f"select=eq(n\\,{index})",
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
        )
        import io

        from PIL import Image

        with Image.open(io.BytesIO(output)) as source:
            return np.asarray(source.convert("RGB")).copy()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(round(index / fps / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            actual = round(float(frame.time) * fps)
            if actual == index:
                return frame.to_ndarray(format="rgb24")
            if actual > index:
                break
    raise ValueError(f"Recorded video does not contain requested RGB frame{index}")


def verify_rgb_video(path: Path, expected_frames: int, expected_sha256: str) -> dict:
    import av

    count, digest = 0, hashlib.sha256()
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            digest.update(frame.to_ndarray(format="rgb24").tobytes())
            count += 1
    if count != expected_frames or digest.hexdigest() != expected_sha256:
        raise ValueError("Recorded RGB video differs from its source pixels or frame count")
    return {"frames": count, "rgb_pixels_identical": True}


def _encode(path: Path, width: int, height: int, fps: int):
    from fractions import Fraction

    import av
    import numpy as np

    if path.exists():
        raise FileExistsError(path)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "bgr0"
        stream.codec_context.gop_size = 1
        stream.codec_context.thread_count = 2
        stream.options = {"level": "3"}
        count, size = 0, width * height * 3
        while True:
            payload = sys.stdin.buffer.read(size)
            if not payload:
                break
            if len(payload) != size:
                raise ValueError("Truncated source RGB frame")
            frame = av.VideoFrame.from_ndarray(
                np.frombuffer(payload, np.uint8).reshape(height, width, 3), format="rgb24"
            )
            frame.pts, frame.time_base = count, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encode", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    _encode(args.encode, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()
