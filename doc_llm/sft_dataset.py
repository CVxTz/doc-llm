import pathlib
from typing import Any, Dict, List

import pandas as pd
import torch
from transformers import DefaultDataCollator, PreTrainedTokenizer

from doc_llm.conversation import get_conv_template
from doc_llm.internvl_utils import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    MAX_IMAGES,
    NUM_IMAGE_TOKEN,
    load_image,
)

PROMPT = "Extract the company, date, address and total in json format. Respond with a valid JSON only."


class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        template: str,
        tokenizer: PreTrainedTokenizer,
        question=PROMPT,
    ):
        self.data = data
        self.template = template
        self.tokenizer = tokenizer
        self.question = question

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, "images"]
        answer = self.data.loc[idx, "entities"]

        pixel_values = load_image(image_path, max_num=MAX_IMAGES).to(torch.bfloat16)
        num_patches = pixel_values.size(0)

        image_bs = pixel_values.shape[0]

        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * NUM_IMAGE_TOKEN * image_bs
            + IMG_END_TOKEN
        )
        question = image_tokens + "\n" + self.question

        template = get_conv_template(self.template)

        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], answer)
        query = template.get_prompt()

        model_inputs = self.tokenizer(query)

        return {
            "input_ids": model_inputs["input_ids"],
            "labels": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "pixel_values": pixel_values,
            "image_flags": [1] * num_patches,
            "query": query,
        }


class CustomDataCollator(DefaultDataCollator):
    def __init__(self, pad_token, ignore_index: int = -100):
        self.pad_token = pad_token
        self.ignore_index = ignore_index

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_flags = []

        max_len = max(len(sample_features["input_ids"]) for sample_features in features)

        for sample_features in features:
            input_ids = sample_features["input_ids"]
            labels = sample_features["labels"]
            attention_mask = sample_features["attention_mask"]

            pixel_values = sample_features["pixel_values"]
            image_flags = sample_features["image_flags"]

            input_ids += [self.pad_token] * (max_len - len(input_ids))
            labels += [self.ignore_index] * (max_len - len(labels))
            attention_mask += [0] * (max_len - len(attention_mask))

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
            batch_pixel_values.append(pixel_values)

            batch_image_flags.extend(image_flags)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.int64),
            "labels": torch.tensor(batch_labels, dtype=torch.int64),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.int64),
            "pixel_values": torch.cat(batch_pixel_values, 0),
            "image_flags": torch.tensor(batch_image_flags, dtype=torch.int64),
        }


def load_data(base_path: pathlib.Path, fold="train"):
    image_base_path = base_path / fold / "img"
    entities_base_path = base_path / fold / "entities"

    images = sorted(list(image_base_path.glob("*.jpg")))
    entities = [(entities_base_path / f"{x.stem}.txt").read_text() for x in images]

    data = pd.DataFrame({"images": images, "entities": entities})

    return data


if __name__ == "__main__":
    from doc_llm.configuration_internvl_chat import InternVLChatConfig
    from doc_llm.tokenization_internlm2 import InternLM2Tokenizer

    data_path = pathlib.Path("/home/youness/Data/SROIE2019")

    path = "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"

    config = InternVLChatConfig.from_pretrained(path)
    _tokenizer = InternLM2Tokenizer.from_pretrained(path)

    patch_size = config.vision_config.patch_size
    image_size = config.force_image_size or config.vision_config.image_size
    num_image_token = int(
        (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
    )

    print(num_image_token)

    print(config)
    print(config.template)

    _data = load_data(data_path, fold="train")

    sft_data = SFTDataset(data=_data, template=config.template, tokenizer=_tokenizer)

    print(sft_data[0])
