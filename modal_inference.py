import os
import urllib

from modal import asgi_app, method, enter, build
from .stub import S3_VIDEO_PATH, MODEL_CACHE, cls_dec, function_dec, volume, stub
from pathlib import Path
# for local testing
#S3_VIDEO_PATH= "s3_videos"
#MODEL_CACHE = "models"
#Path(VOLUME_DIR).mkdir(exist_ok=True, parents=True)
VIDEOS_DIR = Path(S3_VIDEO_PATH) / "videos"
IMAGES_DIR = Path(S3_VIDEO_PATH) / "images"


@cls_dec(gpu="any")
class VideoLlavaModel:
    @build()
    @enter()
    def load_model(self):
        import torch
        from videollava.serve.gradio_utils import Chat
        self.conv_mode = "llava_v1"
        model_path = 'LanguageBind/Video-LLaVA-7B'
        device = 'cuda'
        load_8bit = False
        load_4bit = True
        self.dtype = torch.float16
        self.handler = Chat(model_path, conv_mode=self.conv_mode, load_8bit=load_8bit, load_4bit=load_4bit, device=device, cache_dir=str(MODEL_CACHE))
        print("model loaded")
        # self.handler.model.to(dtype=self.dtype)

    @method()
    def generate(self, image1, video, textbox_in):
        from videollava.conversation import conv_templates
        from videollava.constants import DEFAULT_IMAGE_TOKEN
        if not textbox_in:
            raise ValueError("no prompt provided")

        image1 = image1 if image1 else "none"
        video = video if video else "none"

        state_ = conv_templates[self.conv_mode].copy()
        images_tensor = []

        text_en_in = textbox_in.replace("picture", "image")

        image_processor = self.handler.image_processor
        if os.path.exists(image1) and not os.path.exists(video):
            tensor = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
            tensor = tensor.to(self.handler.model.device, dtype=self.dtype)
            images_tensor.append(tensor)
        video_processor = self.handler.video_processor
        if not os.path.exists(image1) and os.path.exists(video):
            tensor = video_processor(video, return_tensors='pt')['pixel_values'][0]
            tensor = tensor.to(self.handler.model.device, dtype=self.dtype)
            images_tensor.append(tensor)
        if os.path.exists(image1) and os.path.exists(video):
            tensor = video_processor(video, return_tensors='pt')['pixel_values'][0]
            tensor = tensor.to(self.handler.model.device, dtype=self.dtype)
            images_tensor.append(tensor)

            tensor = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
            tensor = tensor.to(self.handler.model.device, dtype=self.dtype)
            images_tensor.append(tensor)

        if os.path.exists(image1) and not os.path.exists(video):
            text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + text_en_in
        elif not os.path.exists(image1) and os.path.exists(video):
            text_en_in = ''.join([DEFAULT_IMAGE_TOKEN] * self.handler.model.get_video_tower().config.num_frames) + '\n' + text_en_in
        elif os.path.exists(image1) and os.path.exists(video):
            text_en_in = ''.join([DEFAULT_IMAGE_TOKEN] * self.handler.model.get_video_tower().config.num_frames) + '\n' + text_en_in + '\n' + DEFAULT_IMAGE_TOKEN
        else:
            print("WARNING: No image or video supplied")

        print(text_en_in)
        text_en_out, _ = self.handler.generate(images_tensor, text_en_in, first_run=True, state=state_)

        text_en_out = text_en_out.split('#')[0]
        textbox_out = text_en_out

        return textbox_out



def fastapi_app():
    from fastapi import FastAPI, UploadFile, File
    import aiofiles

    Path(MODEL_CACHE).mkdir(exist_ok=True, parents=True)
    VIDEOS_DIR.mkdir(exist_ok=True, parents=True)
    IMAGES_DIR.mkdir(exist_ok=True, parents=True)

    app = FastAPI()
    model = VideoLlavaModel()

    @app.post("/upload")
    async def upload(
        file: UploadFile = File(...),
    ):
        filename_decoded = urllib.parse.unquote(file.filename)
        file_path = str(VIDEOS_DIR / filename_decoded)
        async with aiofiles.open(file_path, "wb") as buffer:
            while content := await file.read(1024):  # Read chunks of 1024 bytes
                await buffer.write(content)
        volume.commit()
        return {"file_path": file_path}

    @app.post("/inference")
    async def inference(
        video_file_name: str = '',
        video_file_path: str = '',
        image_file_name: str = '',
        image_file_path: str = '',
        prompt: str = '',
    ):
        import requests
        requests.get('https://huggingface.co/LanguageBind/Video-LLaVA-7B/resolve/main/config.json').raise_for_status()
        video_file_name = urllib.parse.unquote(video_file_name)
        video_file_path = urllib.parse.unquote(video_file_path)
        if video_file_path is None or video_file_path == '':
            if video_file_name is None or video_file_name == '':
                raise ValueError("one of video_file_path or video_file_name must be specified")
            video_file_path = str(VIDEOS_DIR / video_file_name)

        image_file_name = urllib.parse.unquote(image_file_name)
        image_file_path = urllib.parse.unquote(image_file_path)
        if image_file_path is None or image_file_path == '':
            if image_file_name is not None and image_file_name != '':
                image_file_path = str(IMAGES_DIR / image_file_name)

        return model.generate.remote(image_file_path, video_file_path, prompt)
    return app


@function_dec()
@asgi_app()
def fastapi_app_modal():
    return fastapi_app()

# local testing:
# comment this out to deploy
# app = fastapi_app()
# conda activate videollava
# uvicorn modal_inference:app
