from pathlib import Path

import torch
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from doc_llm.internvl_utils import IMG_CONTEXT_TOKEN, wrap_lora
from doc_llm.modeling_internvl_chat import InternVLChatModel
from doc_llm.sft_dataset import CustomDataCollator, SFTDataset, load_data
from doc_llm.tokenization_internlm2 import InternLM2Tokenizer

EPOCHS = 3


if __name__ == "__main__":
    path = "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"

    BASE_PATH = Path(__file__).parents[1] / "outputs"
    refined_model = str(BASE_PATH / "Mini-InternVL-Chat-2B-V1-5-LoRA")

    data_path = Path("/home/youness/Data/SROIE2019")

    _data = load_data(data_path, fold="train")

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = InternVLChatModel.from_pretrained(
        path,
        device_map={"": 0},
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = InternLM2Tokenizer.from_pretrained(path)

    # set the max number of tiles in `max_num`
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    print("img_context_token_id", img_context_token_id)
    model.img_context_token_id = img_context_token_id

    model.config.llm_config.use_cache = False

    model = wrap_lora(model, r=128, lora_alpha=256)

    training_data = SFTDataset(
        data=_data, template=model.config.template, tokenizer=tokenizer
    )

    collator = CustomDataCollator(pad_token=tokenizer.pad_token_id, ignore_index=-100)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    print("img_context_token_id", img_context_token_id)
    model.img_context_token_id = img_context_token_id
    print("model.img_context_token_id", model.img_context_token_id)

    train_params = TrainingArguments(
        output_dir=str(BASE_PATH / "results_modified"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        save_steps=len(training_data) // 10,
        logging_steps=len(training_data) // 50,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.001,
        max_steps=-1,
        group_by_length=False,
        max_grad_norm=1.0,
    )
    # Trainer
    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=training_data,
        dataset_text_field="###",
        tokenizer=tokenizer,
        args=train_params,
        data_collator=collator,
        max_seq_length=tokenizer.model_max_length,
    )

    print(fine_tuning.model.print_trainable_parameters())
    # Training
    fine_tuning.train()
    # Save Model
    fine_tuning.model.save_pretrained(refined_model)
