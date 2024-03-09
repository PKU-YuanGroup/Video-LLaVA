import os
import shutil
import urllib

from modal import asgi_app, method, enter, build
from ai_video_editor.utils.fs_utils import async_copy_from_s3
from .image import LOCAL_VOLUME_DIR, MODEL_CACHE, cls_dec, function_dec, local_volume
from ai_video_editor.stub import stub, S3_VIDEO_PATH, VOLUME_DIR, volume as remote_volume
import diskcache as dc
from pathlib import Path
# for local testing
#S3_VIDEO_PATH= "s3_videos"
#MODEL_CACHE = "models"
#Path(VOLUME_DIR).mkdir(exist_ok=True, parents=True)
VIDEOS_DIR = Path(S3_VIDEO_PATH) / "videos"
IMAGES_DIR = Path(S3_VIDEO_PATH) / "images"



@cls_dec(gpu="any")
class VideoLlavaModel:
    @enter()
    def load_model(self):
        self.cache = dc.Cache('.cache')
        local_volume.reload()
        import torch
        from videollava.serve.gradio_utils import Chat
        self.conv_mode = "llava_v1"
        model_path = 'LanguageBind/Video-LLaVA-7B'
        device = 'cuda'
        load_8bit = False
        load_4bit = True
        self.dtype = torch.float16
        self.handler = Chat(model_path, conv_mode=self.conv_mode, load_8bit=load_8bit, load_4bit=load_4bit, device=device, cache_dir=str(MODEL_CACHE))
        # self.handler.model.to(dtype=self.dtype)

    def copy_file_from_remote_volume(self, filepath):
        in_volume_path = filepath.split('/', 2)[-1]
        local_volume_path = Path(LOCAL_VOLUME_DIR) / in_volume_path
        local_volume_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_volume_path.exists():
            shutil.copy(filepath, str(local_volume_path))

    async def copy_file_from_s3(self, filepath):
        bucket, in_bucket_path = filepath.replace('s3://','').split('/', 1)
        await async_copy_from_s3(bucket, in_bucket_path, str(Path(VOLUME_DIR) / in_bucket_path))

    async def copy_file_to_local(self, filepath):
        if not filepath:
            return
        if filepath.startswith('s3://'):
            await self.copy_file_from_s3(filepath)
        else:
            self.copy_file_from_remote_volume(filepath)

    @method()
    async def generate(self, image1, video, textbox_in, use_existing_output=True):
        inputs = (image1, video, textbox_in)
        if inputs in self.cache and use_existing_output:
            res = self.cache[inputs]
            self.cache.close()
            return res
        remote_volume.reload()
        local_volume.reload()
        await self.copy_file_to_local(image1)
        await self.copy_file_to_local(video)

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

        self.cache.set(inputs, textbox_out)
        self.cache.close()
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
        local_volume.reload()
        filename_decoded = urllib.parse.unquote(file.filename)
        file_path = str(Path(LOCAL_VOLUME_DIR) / filename_decoded)
        async with aiofiles.open(file_path, "wb") as buffer:
            while content := await file.read(1024):  # Read chunks of 1024 bytes
                await buffer.write(content)
        local_volume.commit()
        return {"file_path": file_path}

    @app.post("/inference")
    async def inference(
        video_file_name: str = '',
        video_file_path: str = '',
        image_file_name: str = '',
        image_file_path: str = '',
        prompt: str = '',
    ):
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
