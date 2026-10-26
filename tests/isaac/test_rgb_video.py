"""Exact RGB roundtrip is required before replacing millions of PNG files."""

import shutil

import numpy as np
import pytest

from vla_lego.pretrained.rgb_video import RGBVideoWriter, read_rgb_frame


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="System codec unavailable")
def test_lossless_rgb_video_preserves_color_pixels_and_frame_identity(tmp_path):
    frames = np.random.default_rng(41).integers(0, 256, (3, 16, 24, 3), dtype=np.uint8)
    writer = RGBVideoWriter(tmp_path / "head.mkv", 24, 16)
    for frame in frames:
        writer.write(frame)
    writer.close()
    for index in (2, 0, 1):
        assert np.array_equal(read_rgb_frame(tmp_path / "head.mkv", index), frames[index])
