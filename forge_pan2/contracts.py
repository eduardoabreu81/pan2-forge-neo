"""Generation contracts independent of Forge, CUDA and model libraries."""

# frames follow 4n + 1 (the VAE compresses time 4x after the first frame); sizes are multiples of 32 (16x VAE, 2x2
# patches); the model was trained at 704 x 1248 and 24 fps, up to 481 frames (20 s)
FRAME_STEP = 4
FPS = 24
SIZE_STEP = 32
MAX_FRAMES = 481


class PAN2Error(RuntimeError):
    """An actionable configuration or generation failure."""


# Forge logs and swallows exceptions raised in script callbacks; a rejected PAN 2 request is kept here and raised
# again from the model, which Forge calls directly
_pending_error = None


def set_pending_error(error):
    global _pending_error
    _pending_error = error


def raise_pending_error():
    if _pending_error is not None:
        raise _pending_error


def duration(frames: int, fps: float = FPS) -> float:
    """The clip's length in seconds, as PAN 2 counts it: from the first frame to the last."""
    return (int(frames) - 1) / float(fps)


def check_request(frames: int, width: int, height: int) -> None:
    if (int(frames) - 1) % FRAME_STEP:
        raise PAN2Error(f"PAN 2 needs 4n + 1 frames (241 is 10 s at 24 fps); got {frames}.")
    if int(frames) > MAX_FRAMES:
        raise PAN2Error(f"PAN 2 makes clips of up to {MAX_FRAMES} frames (20 s at 24 fps); got {frames}.")
    if int(width) % SIZE_STEP or int(height) % SIZE_STEP:
        raise PAN2Error(f"PAN 2 needs a width and height divisible by {SIZE_STEP} (it was trained at 1248 x 704); "
                        f"got {width} x {height}.")
