from diffusers import StableDiffusionPipeline
import base64
import io
import clip
from PIL import Image
import torch
from fastapi.testclient import TestClient
from fastapi import FastAPI, UploadFile, File
import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


app = FastAPI()

# Load models once and use them globally
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4").to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)


# Ensure the outputs directory exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")


@app.post("/generate")
async def generate_image(prompt: str):
    image = pipe(prompt).images[0]
    image_path = f"outputs/generated_image.png"  # ✅ Save inside outputs/
    image.save(image_path)

    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    return {"image": encoded_string}



@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(
            clip.tokenize(["a photo of something"]).to(device))
        image_features = clip_model.encode_image(image_input)

    similarity = torch.cosine_similarity(text_features, image_features).item()

    return {"similarity_score": similarity}
