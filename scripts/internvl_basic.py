import argparse
import pathlib

import torch
from transformers import BitsAndBytesConfig

from doc_llm.internvl_utils import load_image
from doc_llm.modeling_internvl_chat import InternVLChatModel
from doc_llm.tokenization_internlm2 import InternLM2Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple prediction example")

    parser.add_argument(
        "--path",
        type=str,
        default="OpenGVLab/Mini-InternVL-Chat-2B-V1-5",
        help="Path to the model or data",
    )
    parser.add_argument(
        "--fold", type=str, default="train", help="Specify the fold to use"
    )
    parser.add_argument(
        "--quant", action="store_true", default=False, help="Enable quantization"
    )
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        default=pathlib.Path("~/Data/SROIE2019").expanduser(),
        help="Path to the data directory",
    )

    args = parser.parse_args()

    image_base_path = args.data_path / args.fold / "img"
    entities_base_path = args.data_path / args.fold / "entities"

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = InternVLChatModel.from_pretrained(
        args.path,
        device_map={"": 0},
        quantization_config=quant_config if args.quant else None,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = InternLM2Tokenizer.from_pretrained(args.path)
    # set the max number of tiles in `max_num`

    model.eval()

    pixel_values = (
        load_image(image_base_path / "X51005255805.jpg", max_num=6)
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
    # print(model)
    response = model.chat(tokenizer, pixel_values, question, generation_config)

    print(response)
