
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_name = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

image = Image.open('noise.png').convert("RGB")
# image = Image.open('biden_resized.png').convert("RGB")
import numpy as np
image = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).to('cuda')

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "You are an image classifier. What is this picture? Print only a noun."},
        ],
    }
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(
    text=[text_prompt], images=image, padding=True, return_tensors="pt"
)

inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)