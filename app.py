import cv2
import torch
import runpod
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from torchvision import transforms


dev = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"using device {dev}")
device = torch.device(dev)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
model = StableDiffusionControlNetPipeline.from_pretrained(
    "XpucT/Deliberate",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16).to(device)
model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
# https://github.com/huggingface/diffusers/issues/2907
model.enable_model_cpu_offload()
model.enable_xformers_memory_efficient_attention()


def inference(event):
    global model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse out your arguments
    model_inputs = event['input']
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', None)
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    image_data = model_inputs.get('image_data', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
    output = model(prompt, image, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps)

    image = output.images[0]
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}

runpod.serverless.start({"handler": inference})
