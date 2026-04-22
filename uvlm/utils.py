import os
import random
import sys


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return os.path.isdir("/content")


def set_seed(seed):
    """Set random seed for reproducibility (random, numpy, torch, cuda, cudnn)."""
    if seed is None:
        return

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_truncation(generated_tokens: int, max_tokens: int) -> tuple:
    """Return (is_truncated: bool, token_count: int)."""
    return generated_tokens >= max_tokens, generated_tokens


def get_hf_token(token=None):
    """
    Get HuggingFace token from multiple sources (in priority order):
    1. Explicit argument
    2. Colab secrets (if in Colab)
    3. HF_TOKEN environment variable
    4. huggingface-cli login cache
    Return None if not found. Never crash.
    """
    if token is not None:
        return token

    if is_colab():
        try:
            from google.colab import userdata
            t = userdata.get("HF_TOKEN")
            if t:
                return t
        except Exception:
            pass

    t = os.environ.get("HF_TOKEN")
    if t:
        return t

    try:
        from huggingface_hub import HfFolder
        t = HfFolder.get_token()
        if t:
            return t
    except Exception:
        pass

    return None
