from modal import Volume, Image, Mount
import os
from pathlib import Path
from ai_video_editor.stub import stub, REPO_HOME, LOCAL_CERT_PATH, CERT_PATH, EXTRA_ENV

LOCAL_VOLUME_DIR = "/video_llava_volume"
HF_DATASETS_CACHE = str(Path(LOCAL_VOLUME_DIR) / "hf_datasets_cache")
MODEL_CACHE = Path(LOCAL_VOLUME_DIR, "models")

LOCAL_VOLUME_NAME = "video-llava-volume"
local_volume = Volume.from_name(LOCAL_VOLUME_NAME, create_if_missing=True)
local_volumes = {
    LOCAL_VOLUME_DIR: local_volume,
}
local_mounts = [
    Mount.from_local_dir("./ai_video_editor/video_llava", remote_path=REPO_HOME),
]


def remove_old_files():
    import shutil
    shutil.rmtree('/volume/models', ignore_errors=True)

image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git",
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "clang",
        "libopenmpi-dev",
        gpu="any",
    )

    .pip_install(
        #  "torch==2.1.2",
        #  "transformers==4.37.2",
        #  "bitsandbytes==0.42.0",
    "torch==2.0.1", "torchvision==0.15.2",
    "transformers==4.31.0", "tokenizers>=0.12.1,<0.14", "sentencepiece==0.1.99", "shortuuid",
    "accelerate==0.21.0", "peft==0.4.0", "bitsandbytes==0.41.0",
    "pydantic<2,>=1", "markdown2[all]", "numpy", "scikit-learn==1.2.2",
    "requests", "httpx==0.24.0", "uvicorn", "fastapi",
    "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13",
    "tensorboardX==2.6.2.2", "gradio==3.37.0", "gradio_client==0.7.0",
    "deepspeed==0.9.5", "ninja", "wandb",
        "wheel",
        gpu="any",
    )
    .run_commands(
        "python -m bitsandbytes",
        gpu="any"
    )
    .run_commands("pip install flash-attn --no-build-isolation", gpu="any")
    .env({"PYTHONPATH": REPO_HOME, "HF_DATASETS_CACHE": HF_DATASETS_CACHE})
    .pip_install(
        "decord",
        "opencv-python",
        "git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d",
        gpu="any",
    )
    .pip_install(
        "aiofiles",
        "aioboto3",
    )
    .run_function(remove_old_files)
    .copy_local_file(LOCAL_CERT_PATH, CERT_PATH)
    .pip_install("boto3", "aioboto3")
    .env(EXTRA_ENV)
    .pip_install("diskcache")
)
# TODO bitsandbytes seems to not be working with gpu

def function_dec(**extras):
    return stub.function(
        image=image,
        timeout=80000,
        # checkpointing doesn't work because it restricts internet access
        #checkpointing_enabled=True,  # Enable memory checkpointing for faster cold starts.
        _allow_background_volume_commits=True,
        container_idle_timeout=120,
        volumes=local_volumes,
        mounts=local_mounts,
        **extras,
    )

def cls_dec(**extras):
    return stub.cls(
        image=image,
        timeout=80000,
        # checkpointing doesn't work because it restricts internet access
        #checkpointing_enabled=True,  # Enable memory checkpointing for faster cold starts.
        container_idle_timeout=120,
        _allow_background_volume_commits=True,
        volumes=local_volumes,
        mounts=local_mounts,
        **extras,
    )
