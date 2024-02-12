from modal import Volume, Image, Stub, Mount, Secret
from pathlib import Path
REPO_HOME = "/app"
VOLUME_DIR = "/volume"
MODELS_DIR = "/root"
HF_DATASETS_CACHE = str(Path(VOLUME_DIR) / "hf_datasets_cache")
MODEL_CACHE = Path(VOLUME_DIR, "models")
assets_path = Path(__file__).parent /  "assets"
local_examples_path = Path(__file__).parent /  "videollava" / "serve" / "examples"
EXAMPLES_PATH = "/examples"
mounts = [
    Mount.from_local_dir("./ai_video_editor/updated_video_llava", remote_path=REPO_HOME),
    Mount.from_local_dir(assets_path, remote_path="/assets"),
    Mount.from_local_dir(local_examples_path, remote_path=EXAMPLES_PATH),
]
volume = Volume.persisted("video-llava-vol")
volumes = {VOLUME_DIR: volume}
stub = Stub("updated-video-llava", mounts=mounts, volumes=volumes, secrets=[Secret.from_dotenv()])


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
    )
    .run_function(remove_old_files)
)
# TODO bitsandbytes seems to not be working with gpu

def function_dec(**extras):
    return stub.function(
        image=image,
        timeout=80000,
        # checkpointing doesn't work because it restricts internet access
        #checkpointing_enabled=True,  # Enable memory checkpointing for faster cold starts.
        _allow_background_volume_commits=True,
        **extras,
    )

def cls_dec(**extras):
    return stub.cls(
        image=image,
        timeout=80000,
        # checkpointing doesn't work because it restricts internet access
        #checkpointing_enabled=True,  # Enable memory checkpointing for faster cold starts.
        **extras,
    )
