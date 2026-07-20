"""Named configs. Import these instead of hand-rolling hyperparameters."""

from model.arch import TernaryDiffusionConfig


def config_300m(vocab_size: int = 32001, mask_token_id: int = 32000) -> TernaryDiffusionConfig:
    """~340M total, ~308M ternary (non-embedding). The kill-switch target.

    Sized to fit comfortably on a single A100/H100 at seq_len=2048 with bf16 +
    activation checkpointing. To land nearer 300M total, drop to n_layers=22 or
    d_ff=2560.
    """
    return TernaryDiffusionConfig(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=2816,
        max_seq_len=2048,
    )


def config_smoke(vocab_size: int = 512, mask_token_id: int = 511) -> TernaryDiffusionConfig:
    """Tiny. Runs on CPU or Apple MPS in seconds. For verifying the diffusion+ternary
    combination does not NaN at step 1, before spending a cent on RunPod."""
    return TernaryDiffusionConfig(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        max_seq_len=64,
    )


def config_mid(vocab_size: int = 16001, mask_token_id: int = 16000) -> TernaryDiffusionConfig:
    """~27M total / ~20M ternary. The intermediate rung between config_local (toy) and
    config_300m (kill-switch). Sized to the empirical ceiling of an Apple-MPS Mac:
    above ~30M, MPS unified memory thrashes and throughput collapses ~70x (measured).
    On a real GPU you would push this to 50-100M; on MPS this is the largest that
    sustains usable speed (~5k tok/s). Use to check the ternary≈FP16 parity above the
    toy regime before betting on the 300M A100 run."""
    return TernaryDiffusionConfig(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        d_model=448,
        n_layers=8,
        n_heads=7,
        d_ff=1216,
        max_seq_len=512,
    )


def config_local(vocab_size: int = 8001, mask_token_id: int = 8000) -> TernaryDiffusionConfig:
    """~7M params. Bigger than smoke, still trainable on Apple MPS in minutes. For a
    real-corpus ternary-vs-FP16 sanity well above the toy-noise floor — NOT the
    kill-switch experiment (that is config_300m on a GPU)."""
    return TernaryDiffusionConfig(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        d_model=256,
        n_layers=6,
        n_heads=4,
        d_ff=768,
        max_seq_len=512,
    )
