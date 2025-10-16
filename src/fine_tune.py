import os

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_DATASET = os.getenv("HF_DATASET")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME")


def main():
    max_seq_length = 2048
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = (
        True  # Use 4bit quantization to reduce memory usage. Can be False.
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Add LoRa adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Any number > 0. Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
    )

    # Load dataset
    dataset = load_dataset(HF_DATASET, split="train")

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama",
        map_eos_token=True,  # Maps <|im_end|> to </s> instead
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    pass

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Train model
    trainer.train()

    # Merge model, 4bit quantization
    model.save_pretrained_gguf(
        MODEL_NAME, tokenizer, quantization_method="q4_k_m"
    )
    model.push_to_hub_gguf(
        HF_MODEL_NAME, tokenizer, quantization_method="q4_k_m"
    )


if __name__ == "__main__":
    main()
