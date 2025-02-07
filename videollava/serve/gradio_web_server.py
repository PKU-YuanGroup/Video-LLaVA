#  import shutil
#  import os
#  import tempfile

#  from modal import asgi_app, method, enter
#  from ...stub import VOLUME_DIR, MODEL_CACHE, cls_dec, function_dec


#  def save_image_to_local(image):
    #  from PIL import Image
    #  filename = os.path.join(VOLUME_DIR, next(tempfile._get_candidate_names()) + '.jpg')
    #  image = Image.open(image)
    #  image.save(filename)
    #  # print(filename)
    #  return filename


#  def save_video_to_local(video_path):
    #  filename = os.path.join(VOLUME_DIR, next(tempfile._get_candidate_names()) + '.mp4')
    #  shutil.copyfile(video_path, filename)
    #  return filename


#  @cls_dec(gpu="any")
#  class VideoLlavaModel:
    #  @enter()
    #  def load_model(self):
        #  import torch
        #  from videollava.serve.gradio_utils import Chat
        #  self.conv_mode = "llava_v1"
        #  model_path = 'LanguageBind/Video-LLaVA-7B'
        #  device = 'cuda'
        #  load_8bit = True
        #  load_4bit = False
        #  self.dtype = torch.float16
        #  self.handler = Chat(model_path, conv_mode=self.conv_mode, load_8bit=load_8bit, load_4bit=load_4bit, device=device, cache_dir=MODEL_CACHE)
        #  # self.handler.model.to(dtype=self.dtype)

    #  @method()
    #  def generate(self, image1, video, textbox_in, first_run, state, state_, images_tensor):
        #  from videollava.conversation import conv_templates, Conversation
        #  import gradio as gr
        #  from videollava.constants import DEFAULT_IMAGE_TOKEN
        #  flag = 1
        #  if not textbox_in:
            #  if len(state_.messages) > 0:
                #  textbox_in = state_.messages[-1][1]
                #  state_.messages.pop(-1)
                #  flag = 0
            #  else:
                #  return "Please enter instruction"

        #  image1 = image1 if image1 else "none"
        #  video = video if video else "none"
        #  # assert not (os.path.exists(image1) and os.path.exists(video))

        #  if type(state) is not Conversation:
            #  state = conv_templates[self.conv_mode].copy()
            #  state_ = conv_templates[self.conv_mode].copy()
            #  images_tensor = []

        #  first_run = False if len(state.messages) > 0 else True

        #  text_en_in = textbox_in.replace("picture", "image")

        #  # images_tensor = [[], []]
        #  image_processor = self.handler.image_processor
        #  if os.path.exists(image1) and not os.path.exists(video):
            #  tensor = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
            #  # print(tensor.shape)
            #  tensor = tensor.to(self.handler.model.device, dtype=self.dtype)
            #  images_tensor.append(tensor)
        #  video_processor = self.handler.video_processor
        #  if not os.path.exists(image1) and os.path.exists(video):
            #  tensor = video_processor(video, return_tensors='pt')['pixel_values'][0]
            #  # print(tensor.shape)
            #  tensor = tensor.to(self.handler.model.device, dtype=self.dtype)
            #  images_tensor.append(tensor)
        #  if os.path.exists(image1) and os.path.exists(video):
            #  tensor = video_processor(video, return_tensors='pt')['pixel_values'][0]
            #  # print(tensor.shape)
            #  tensor = tensor.to(self.handler.model.device, dtype=self.dtype)
            #  images_tensor.append(tensor)

            #  tensor = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
            #  # print(tensor.shape)
            #  tensor = tensor.to(self.handler.model.device, dtype=self.dtype)
            #  images_tensor.append(tensor)

        #  if os.path.exists(image1) and not os.path.exists(video):
            #  text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + text_en_in
        #  if not os.path.exists(image1) and os.path.exists(video):
            #  text_en_in = ''.join([DEFAULT_IMAGE_TOKEN] * self.handler.model.get_video_tower().config.num_frames) + '\n' + text_en_in
        #  if os.path.exists(image1) and os.path.exists(video):
            #  text_en_in = ''.join([DEFAULT_IMAGE_TOKEN] * self.handler.model.get_video_tower().config.num_frames) + '\n' + text_en_in + '\n' + DEFAULT_IMAGE_TOKEN
        #  # print(text_en_in)
        #  text_en_out, state_ = self.handler.generate(images_tensor, text_en_in, first_run=first_run, state=state_)
        #  state_.messages[-1] = (state_.roles[1], text_en_out)

        #  text_en_out = text_en_out.split('#')[0]
        #  textbox_out = text_en_out

        #  show_images = ""
        #  if os.path.exists(image1):
            #  filename = save_image_to_local(image1)
            #  show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
        #  if os.path.exists(video):
            #  filename = save_video_to_local(video)
            #  show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'

        #  if flag:
            #  state.append_message(state.roles[0], textbox_in + "\n" + show_images)
        #  state.append_message(state.roles[1], textbox_out)

        #  return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=image1 if os.path.exists(image1) else None, interactive=True), gr.update(value=video if os.path.exists(video) else None, interactive=True))

    #  @method()
    #  def clear_history(self, state, state_):
        #  from videollava.conversation import conv_templates
        #  import gradio as gr
        #  state = conv_templates[self.conv_mode].copy()
        #  state_ = conv_templates[self.conv_mode].copy()
        #  return (gr.update(value=None, interactive=True),
                #  gr.update(value=None, interactive=True), \
                #  gr.update(value=None, interactive=True), \
                #  True, state, state_, state.to_gradio_chatbot(), [])




#  def regenerate(state, state_):
    #  state.messages.pop(-1)
    #  state_.messages.pop(-1)
    #  if len(state.messages) > 0:
        #  return state, state_, state.to_gradio_chatbot(), False
    #  return (state, state_, state.to_gradio_chatbot(), True)





#  def build_gradio_interface(model):
    #  import gradio as gr
    #  from videollava.serve.gradio_utils import tos_markdown, learn_more_markdown, title_markdown, block_css

    #  #  if not os.path.exists("temp"):
        #  #  os.makedirs("temp")


    #  textbox = gr.Textbox(
        #  show_label=False, placeholder="Enter text and press ENTER", container=False
    #  )
    #  with gr.Blocks(title='Video-LLaVA🚀', theme=gr.themes.Default(), css=block_css) as interface:
        #  gr.Markdown(title_markdown)
        #  state = gr.State()
        #  state_ = gr.State()
        #  first_run = gr.State()
        #  images_tensor = gr.State()

        #  with gr.Row():
            #  with gr.Column(scale=3):
                #  image1 = gr.Image(label="Input Image", type="filepath")
                #  video = gr.Video(label="Input Video")

                #  cur_dir = os.path.dirname(os.path.abspath(__file__))
                #  gr.Examples(
                    #  examples=[
                        #  [
                            #  f"{cur_dir}/examples/extreme_ironing.jpg",
                            #  "What is unusual about this image?",
                        #  ],
                        #  [
                            #  f"{cur_dir}/examples/waterview.jpg",
                            #  "What are the things I should be cautious about when I visit here?",
                        #  ],
                        #  [
                            #  f"{cur_dir}/examples/desert.jpg",
                            #  "If there are factual errors in the questions, point it out; if not, proceed answering the question. What’s happening in the desert?",
                        #  ],
                    #  ],
                    #  inputs=[image1, textbox],
                #  )

            #  with gr.Column(scale=7):
                #  chatbot = gr.Chatbot(label="Video-LLaVA", bubble_full_width=True).style(height=750)
                #  with gr.Row():
                    #  with gr.Column(scale=8):
                        #  textbox.render()
                    #  with gr.Column(scale=1, min_width=50):
                        #  submit_btn = gr.Button(
                            #  value="Send", variant="primary", interactive=True
                        #  )
                #  with gr.Row(elem_id="buttons") as button_row:
                    #  upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                    #  downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                    #  flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                    #  # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    #  regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                    #  clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

        #  with gr.Row():
            #  gr.Examples(
                #  examples=[
                    #  [
                        #  f"{cur_dir}/examples/sample_img_22.png",
                        #  f"{cur_dir}/examples/sample_demo_22.mp4",
                        #  "Are the instruments in the pictures used in the video?",
                    #  ],
                    #  [
                        #  f"{cur_dir}/examples/sample_img_13.png",
                        #  f"{cur_dir}/examples/sample_demo_13.mp4",
                        #  "Does the flag in the image appear in the video?",
                    #  ],
                    #  [
                        #  f"{cur_dir}/examples/sample_img_8.png",
                        #  f"{cur_dir}/examples/sample_demo_8.mp4",
                        #  "Are the image and the video depicting the same place?",
                    #  ],
                #  ],
                #  inputs=[image1, video, textbox],
            #  )
            #  gr.Examples(
                #  examples=[
                    #  [
                        #  f"{cur_dir}/examples/sample_demo_1.mp4",
                        #  "Why is this video funny?",
                    #  ],
                    #  [
                        #  f"{cur_dir}/examples/sample_demo_3.mp4",
                        #  "Can you identify any safety hazards in this video?"
                    #  ],
                    #  [
                        #  f"{cur_dir}/examples/sample_demo_9.mp4",
                        #  "Describe the video.",
                    #  ],
                    #  [
                        #  f"{cur_dir}/examples/sample_demo_22.mp4",
                        #  "Describe the activity in the video.",
                    #  ],
                #  ],
                #  inputs=[video, textbox],
            #  )
        #  gr.Markdown(tos_markdown)
        #  gr.Markdown(learn_more_markdown)

        #  submit_btn.click(model.generate.remote, [image1, video, textbox, first_run, state, state_, images_tensor],
                         #  [state, state_, chatbot, first_run, textbox, images_tensor, image1, video])

        #  regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
            #  model.generate.remote, [image1, video, textbox, first_run, state, state_, images_tensor], [state, state_, chatbot, first_run, textbox, images_tensor, image1, video])

        #  clear_btn.click(model.clear_history.remote, [state, state_],
                        #  [image1, video, textbox, first_run, state, state_, chatbot, images_tensor])
    #  return interface


#  @function_dec(gpu="any")
#  @asgi_app()
#  def fastapi_app():
    #  from gradio.routes import mount_gradio_app
    #  from fastapi import FastAPI
    #  app = FastAPI()

    #  model = VideoLlavaModel()
    #  #  interface = gr.Interface(
        #  #  fn=classifier.predict.remote,
        #  #  inputs=gr.Image(shape=(224, 224)),
        #  #  outputs="label",
        #  #  examples=create_demo_examples(),
        #  #  css="/assets/index.css",
    #  #  )
    #  return mount_gradio_app(
        #  app=app,
        #  blocks=build_gradio_interface(model),
        #  path="/",
    #  )
#  # app = gr.mount_gradio_app(app, demo, path="/")
#  # demo.launch()

#  # uvicorn videollava.serve.gradio_web_server:app
#  # python -m  videollava.serve.gradio_web_server
