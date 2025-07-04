from fastapi import FastAPI, File, UploadFile
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re
from PIL import Image
import json
import io

# Initialize the app
app = FastAPI()

# Load the model and processor (this happens only once when the server starts)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.post("/process-receipt/")
async def process_receipt(file: UploadFile = File(...)):
    # 1. Read the image from the uploaded file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # 2. Process the image using the Donut model (your script's logic)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    # 3. Return the structured JSON
    return processor.token2json(sequence)