import argparse
import pathlib
from statistics import mean

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from doc_llm.evaluation_utils import fuzz_score_dicts, parse_json
from doc_llm.internvl_utils import load_image
from doc_llm.modeling_internvl_chat import InternVLChatModel
from doc_llm.sft_dataset import PROMPT
from doc_llm.tokenization_internlm2 import InternLM2Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Eval")

    parser.add_argument(
        "--path",
        type=str,
        default="OpenGVLab/Mini-InternVL-Chat-2B-V1-5",
        help="Path to the model or data",
    )
    parser.add_argument(
        "--fold", type=str, default="test", help="Specify the fold to use"
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

    scores = []

    for img in tqdm(sorted(list(image_base_path.glob("*.jpg")))):
        stem = img.stem
        entities_path = entities_base_path / f"{stem}.txt"

        gt_output = entities_path.read_text()

        pixel_values = load_image(img, max_num=6).to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        # single-round single-image conversation
        question = PROMPT
        mllm_output = model.chat(tokenizer, pixel_values, question, generation_config)

        _gt = parse_json(gt_output)
        _prediction = parse_json(mllm_output)

        fuzz_score = fuzz_score_dicts(_gt, _prediction)

        scores.append(fuzz_score)

        print(stem, fuzz_score)

    print(f"Average score: {mean(scores)}")

    # OpenGVLab/Mini-InternVL-Chat-2B-V1-5 : Average score: 74.2478386167147
    # OpenGVLab/Mini-InternVL-Chat-2B-V1-5-LoRA : Average score: 95.4
