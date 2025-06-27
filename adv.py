from PIL import Image
import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained('./tmp', local_files_only=True, torch_dtype="auto", device_map="auto").eval()
processor = AutoProcessor.from_pretrained('./tmp', local_files_only=True, torch_dtype="auto", device_map="auto")

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# input_tgt = processor.tokenizer('dog', return_tensors='pt').input_ids[0]
# print(input_tgt)

image = Image.open('biden_resized.png').convert("RGB")

import numpy as np
image = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).to('cuda')
image.requires_grad = True

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
tmp_img_input = inputs['pixel_values']
tmp_img_input_2 = tmp_img_input.view(-1, 2, 14, 14)
img_grad = tmp_img_input_2[:, 0]
img_grad.requires_grad = True
img_processed = img_grad.unsqueeze(-3).expand(-1, 2, -1, -1).reshape(256, 1176)

inputs['pixel_values'] = img_processed
inputs = inputs.to("cuda")

'''
Parameters from QWEN

channel = 3, 
grid_t = 1, grid_h = 16, grid_w = 16, self.patch_size = 14, 
self.temporal_patch_size=2, self.merge_size=2
'''

'''
SOURCE CODE

channel = patches.shape[1]
grid_t = patches.shape[0] // self.temporal_patch_size
grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
patches = patches.reshape(
    grid_t,
    self.temporal_patch_size,
    channel,
    grid_h // self.merge_size,
    self.merge_size,
    self.patch_size,
    grid_w // self.merge_size,
    self.merge_size,
    self.patch_size,
)
# (1, 2, 3, 8, 2, 14, 8, 2, 14)
patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
# (1, 8, 8, 2, 2, 3, 2, 14, 14)
flatten_patches = patches.reshape(
    grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
)
# (256, 1176)
'''

################ TEST IMAGE RECONSTRUCTION ####################
tmp = tmp_img_input.cpu().numpy()
reshaped_tmp = tmp.reshape(1, 8, 8, 2, 2, 3, 2, 14, 14)
transposed_tmp = reshaped_tmp.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)
dup_tmp = transposed_tmp.reshape(2, 3, 224, 224)
original_tmp = dup_tmp[0]
image_mean = [0.48145466,
    0.4578275,
    0.40821073]
image_std = [
    0.26862954,
    0.26130258,
    0.27577711
]
original_tmp = original_tmp * np.array(image_std)[:, None, None] + np.array(image_mean)[:, None, None]
original_tmp = original_tmp * 255

tmp_image = Image.fromarray(original_tmp.transpose(1,2,0).astype(np.uint8))
tmp_image.save('revert.png')

epsilon = ((0.5/255)*np.ones_like(original_tmp))/np.array(image_std)[:, None, None]
epsilon = epsilon.reshape(1, 1, 3, 8, 2, 14, 8, 2, 14).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
epsilon = epsilon.reshape(-1, 14, 14)
epsilon = torch.from_numpy(epsilon)

min_pixel = (np.zeros_like(original_tmp)-np.array(image_mean)[:, None, None])/np.array(image_std)[:, None, None]
min_pixel = min_pixel.reshape(1, 1, 3, 8, 2, 14, 8, 2, 14).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
min_pixel = min_pixel.reshape(-1, 14, 14)
min_pixel = torch.from_numpy(min_pixel)
max_pixel = (np.ones_like(original_tmp)-np.array(image_mean)[:, None, None])/np.array(image_std)[:, None, None]
max_pixel = max_pixel.reshape(1, 1, 3, 8, 2, 14, 8, 2, 14).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
max_pixel = max_pixel.reshape(-1, 14, 14)
max_pixel = torch.from_numpy(max_pixel)

first_image = img_grad.detach().clone()
best_image = first_image.clone()
best = 15
for i in range(100):
    img_processed = img_grad.unsqueeze(-3).expand(-1, 2, -1, -1).reshape(256, 1176)
    inputs['pixel_values'] = img_processed
    output_ids = model(
        **inputs)
    class_dist = output_ids.logits[0][-1]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(class_dist.unsqueeze(dim=0), torch.tensor([18457]).to(class_dist.device))
    print(loss.item())
    if loss.item()<best:
        best = loss.item()
        best_image = img_grad.detach().clone()
    loss.backward()
    img_grad = img_grad - epsilon*img_grad.grad.detach().sign()
    img_grad = first_image + torch.clamp(img_grad-first_image, -16*epsilon, 16*epsilon)
    img_grad = torch.clamp(img_grad, min_pixel, max_pixel)
    img_grad = img_grad.detach().requires_grad_()

final_tensor = best_image
final_tensor = final_tensor.detach().cpu().numpy()

reshaped_fnl = final_tensor.reshape(1, 8, 8, 2, 2, 3, 1, 14, 14)

transposed_fnl = reshaped_fnl.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)
original_fnl = transposed_fnl.reshape(3, 224, 224)
original_fnl = original_fnl * np.array(image_std)[:, None, None] + np.array(image_mean)[:, None, None]
original_fnl = original_fnl * 255
final_img = Image.fromarray(original_fnl.transpose(1,2,0).astype(np.uint8))
final_img.save('noise.png')

image = Image.open('noise.png').convert("RGB")
image = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).to('cuda')
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