# EraX-VL-7B-V1
<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66e93d483745423cbb14c5ff/fNxjr3en_onzbOv0sghpE.jpeg" alt="Logo" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/erax/EraX-VL-7B-V1">Hugging Face</a>&nbsp&nbsp </a>
<br>
</p>

## Introduction

After a month's relentless efforts, today we are thrilled to release **EraX-VL-7B-V1**! 

`NOTA BENE: EraX-VL-7B-V1 is NOT a typical OCR-only tool likes Tesseract but is a Multimodal LLM-based model. To use it effectively, you may have to twist your prompt carefully depending on your tasks.`

`EraX-VL-7B-V1` is the latest version of the vision language models in the EraX model families. 

## Benchmark
Below is the evaluation benchmark of **global open-source and proprietary Multimodal Models** on the [MTVQA](https://huggingface.co/datasets/ByteDance/MTVQA) Vietnamese test set conducted by [VinBigdata](https://www.linkedin.com/feed/update/urn:li:activity:7243887708966641664/). We plan to conduct more detailed and diverse evaluations in the near future.
<div align="left">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66e93d483745423cbb14c5ff/-OYkSDVyAcAcLLgO2N5XT.jpeg" width="500"/>
  
  <a href="https://www.linkedin.com/feed/update/urn:li:activity:7243887708966641664/" target="_blank"> (Source: VinBigData)</a>
  (20:00 23 September 2024)
  <br>
</div>

## Quickstart
Below, we provide simple examples to show how to use `EraX-VL-7B-V1` 🤗 Transformers.

The code of `EraX-VL-7B-V1` has been in the latest Hugging face transformers and we advise you to build from source with command:

Install the necessary packages:
```bash
python -m pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
python -m pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```
### Using Google Colaboratory
* Google Colaboratory run right away: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnSxtWDLG48-NQh7wk9_z8WI7J4OY_Ci?usp=sharing)
* Google Colaboratory API (key required): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/179ZzegjBmuvr-5QfJfJzbVOzRTgD0K_r?usp=sharing)

### Using 🤗 Transformers

```python
import os
import base64
import json

import cv2
import numpy as np
import matplotlib.pyplot as  plt

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "erax/EraX-VL-7B-V1"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager", # replace with "flash_attention_2" if your GPU is Ampere architecture
    device_map="auto"
)

tokenizer =  AutoTokenizer.from_pretrained(model_path)
# processor = AutoProcessor.from_pretrained(model_path)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
     model_path,
     min_pixels=min_pixels,
     max_pixels=max_pixels,
 )

image_path ="image.jpg"

with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
decoded_image_text = encoded_image.decode('utf-8')
base64_data = f"data:image;base64,{decoded_image_text}"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": base64_data,
            },
            {
                "type": "text",
                "text": "Diễn tả nội dung bức ảnh như 1 bác sỹ giỏi."
                # "Diễn tả nội dung bức ảnh này bằng định dạng json."
            },
        ],
    }
]

# Prepare prompt
tokenized_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[ tokenized_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Generation configs
generation_config                    = model.generation_config
generation_config.do_sample          = True
generation_config.temperature        = 0.2
generation_config.top_k              = 1
generation_config.top_p              = 0.001
generation_config.max_new_tokens     = 2048
generation_config.repetition_penalty = 1.1

# Inference
generated_ids = model.generate(**inputs, generation_config=generation_config)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])
```

## For API inquiry
- For correspondence regarding this work or inquiry for API trial, please contact Nguyễn Anh Nguyên at [nguyen@erax.ai](nguyen@erax.ai).

## Citation
If you find our project useful, we would appreciate it if you could star our repository and cite our work as follows:
```BibTeX
@article{EraX-VL-7B-V1,
  title={EraX-VL-7B-V1: A Highly Efficient Multimodal LLM for Vietnamese, especially for medical forms and bills},
  author={Nguyễn Anh Nguyên and Nguyễn Hồ Nam (BCG) and Hoàng Tiến Dũng and Phạm Đình Thục and Phạm Huỳnh Nhật},
  organization={EraX},
  year={2024},
  url={https://huggingface.co/erax-ai/EraX-VL-7B-V1}
}
```
## Acknowledgement

`EraX-VL-7B-V1` is built with reference to the code of the following projects: [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL), [InternVL](https://github.com/OpenGVLab/InternVL) and Khang Đoàn ([5CD-AI](https://huggingface.co/5CD-AI)). Thanks for their awesome work!
