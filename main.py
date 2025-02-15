from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import clip
import io
import base64
from diffusers import StableDiffusionPipeline

app = FastAPI()

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4").to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

@app.post("/generate")
async def generate_image(prompt: str):
    image = pipe(prompt).images[0]
    image.save("static/generated.png")

    with open("static/generated.png", "rb") as img_file:
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
