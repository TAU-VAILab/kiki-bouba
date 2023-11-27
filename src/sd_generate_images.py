from utils import words
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import uuid
import os
from tqdm.auto import tqdm

def word2prompt(word):
    return f'a 3D rendering of a {word} shaped object'

def main():
    SD_KWARGS = {
        'guidance_scale': 9,
        'num_inference_steps': 20
    }
    N = 50

    MODEL_ID = 'stabilityai/stable-diffusion-2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    OUTPUT_DIR = 'generated_images'
    if not os.path.exists(OUTPUT_DIR):
        print("Making output image directory:", OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    print("Loading model:", MODEL_ID)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    print("Model loaded")

    words_kb = words + ['kiki', 'bouba']

    SEEN = {w for w in words_kb if os.path.exists(f'{OUTPUT_DIR}/{w}')}
    unseen = list(set(words_kb) - SEEN)

    print("Seen words:", len(SEEN))
    print("Unseen words:", len(unseen))

    for word in tqdm(unseen, desc="Generating images"):
        d = f'{OUTPUT_DIR}/{word}'
        if not os.path.exists(d):
            prompt = word2prompt(word)
            out = pipe(
                prompt,
                num_images_per_prompt=N,
                **SD_KWARGS
            )
            os.makedirs(d, exist_ok=True)
            for img in out.images:
                basefn = word + '-' + uuid.uuid4().hex + '.jpg'
                fn = f'{d}/{basefn}'
                img.save(fn)

if __name__ == "__main__":
    main()