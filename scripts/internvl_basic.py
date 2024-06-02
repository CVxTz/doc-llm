import torch
from transformers import BitsAndBytesConfig

from doc_llm.internvl_utils import load_image
from doc_llm.modeling_internvl_chat import InternVLChatModel
from doc_llm.tokenization_internlm2 import InternLM2Tokenizer

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

if __name__ == "__main__":
    path = "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"

    model = InternVLChatModel.from_pretrained(
        path,
        device_map={"": 0},
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = InternLM2Tokenizer.from_pretrained(path)
    # set the max number of tiles in `max_num`

    model.eval()

    pixel_values = (
        load_image("/home/youness/Data/SROIE2019/train/img/X51005255805.jpg", max_num=6)
        .to(torch.bfloat16)
        .cuda()
    )

    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )

    # single-round single-image conversation
    question = (
        "Extract the company, date, address and total in json format."
        "Respond with a valid JSON only."
    )
    print(model)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(response)
