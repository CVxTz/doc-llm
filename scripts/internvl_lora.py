import torch
from transformers import BitsAndBytesConfig

from doc_llm.internvl_utils import load_image, wrap_backbone_lora, wrap_llm_lora
from doc_llm.modeling_internvl_chat import InternVLChatModel
from doc_llm.tokenization_internlm2 import InternLM2Tokenizer

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

path = "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"

model = InternVLChatModel.from_pretrained(
    path,
    device_map={"": 0},
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

tokenizer = InternLM2Tokenizer.from_pretrained(path)
# set the max number of tiles in `max_num`

wrap_backbone_lora(model, r=16, lora_alpha=32)
wrap_llm_lora(model, r=16, lora_alpha=32)

for name, param in model.named_parameters():
    print(name, param.requires_grad)

model.eval()

pixel_values = load_image("image.png", max_num=6).to(torch.bfloat16).cuda()

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

# single-round single-image conversation
question = (
    "Please describe the picture in detail"  # Please describe the picture in detail
)
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(question, response)
