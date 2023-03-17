import argparse
import os
import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(description='Script for Training LoRAs for a Causal Language Model')
    parser.add_argument('--local_model_path', type=str, required=True, help='Path to the local model')
    parser.add_argument('--load-in-8bit', action='store_true', default=False, help='Load the model in 8bit')
    parser.add_argument('--r', type=int, default=16, help='Number of attention heads for LoRA')
    parser.add_argument('--lora_alpha', type=int, default=32,help= 'LoRAs alpha hyperparameter')
    parser.add_argument('--lora_dropout', type=float, default=0.5, help='LoRAs dropout rate')
    parser.add_argument('--data_path', type=str, required=True, help='path to the dataset .txt file')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to the save the LoRA adapter')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Number of steps for the warmup phase')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-7, help='Initial learning rate')
    parser.add_argument('--fp16', action='store_true', default=False, help='Use mixed-precision training')
    parser.add_argument('--logging_steps', type=int, default=20, help='Number of steps between logging')
    parser.add_argument('--output_dir', type=str, help='Directory for storing the model and logs')
    args = parser.parse_args()
    return args



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = AutoModelForCausalLM.from_pretrained(
        args.local_model_path,
        load_in_8bit=args.load_in_8bit,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path)

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)


    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


    config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )


    model = get_peft_model(model,config)

    print_trainable_parameters(model)


    with open(args.data_path, "r") as file:
        raw_data = file.read()

    def tokenize_and_chunk(text, tokenizer, max_length=2048):
        tokens = tokenizer.encode(text, add_special_tokens=False)

        token_chunks = []
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]
            token_chunks.append(chunk)

        return token_chunks

    token_chunks = tokenize_and_chunk(raw_data, tokenizer)
    
    data = Dataset.from_dict({"input_ids": token_chunks})
    data = data.map(lambda samples: {'decoded_text': tokenizer.batch_decode(samples['input_ids'], skip_special_tokens=True)}, batched=True)

    for i in range(5):
        tokenized_sample = tokenizer(data[i]["decoded_text"])
        token_length = len(tokenized_sample["input_ids"])

        print(f"Sample {i}:")
        print(data[i]["decoded_text"])
        print(f"\nToken Length: {token_length}")
        print("\n" + "=" * 50 + "\n")



    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=TrainingArguments(
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_steps = args.warmup_steps,
            num_train_epochs = args.num_train_epochs,
            learning_rate = args.learning_rate,
            fp16 = args.fp16,
            logging_steps = args.logging_steps,
            output_dir = args.output_dir
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False
    trainer.train()

    adapter_path = args.adapter_path
    config.save_pretrained(args.adapter_path)

    model.save_pretrained(args.adapter_path)

    print("Training finished! Have fun!")



