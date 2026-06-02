import torch


class DeviceService:
    """Select a Torch device and default dtype for model execution."""

    @staticmethod
    def get_device_type() -> tuple[torch.device, torch.dtype]:
        """Return the preferred available Torch device and dtype.

        Returns:
            A `(device, dtype)` tuple preferring CUDA, then Apple MPS, then CPU.
        """
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.bfloat16
        elif torch.mps.is_available():
            return torch.device("mps"), torch.float16
        else:
            return torch.device("cpu"), torch.float32
