#!/usr/bin/env python3
"""
LLM训练脚本
"""
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==================== 配置区域 ====================
BASE_MODEL = "Qwen/Qwen3-8B"
DATASET_PATH = "./dataset.jsonl"
OUTPUT_DIR = "./qwen3-asa-qlora"
CONFIG_DIR = "./"

# ==================== 从原项目加载配置 ====================
def load_original_config(config_dir):
    """加载原项目的配置文件"""
    config = {}
    with open(Path(config_dir) / "adapter_config.json", 'r') as f:
        config['lora'] = json.load(f)
    with open(Path(config_dir) / "chat_template.jinja", 'r') as f:
        config['chat_template'] = f.read()
    # 不再加载 tokenizer_config.json，避免潜在冲突
    return config

print("1. 加载原项目配置文件...")
original_config = load_original_config(CONFIG_DIR)

# ==================== 1. 加载并配置Tokenizer ====================
print("2. 加载并配置Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=False,
    padding_side="right",
)
# 安全设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 应用核心的聊天模板
tokenizer.chat_template = original_config['chat_template']

# ==================== 2. 配置4-bit量化 (QLoRA) ====================
print("3. 配置4-bit量化 (QLoRA)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ==================== 3. 加载基座模型 ====================
print("4. 加载基座模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,
    attn_implementation="sdpa",
)

# ==================== 4. 准备模型并应用LoRA ====================
print("5. 准备模型并应用LoRA配置...")
model = prepare_model_for_kbit_training(model)

lora_cfg = original_config['lora']
peft_config = LoraConfig(
    r=lora_cfg['r'],
    lora_alpha=lora_cfg['lora_alpha'],
    lora_dropout=lora_cfg['lora_dropout'],
    target_modules=lora_cfg['target_modules'],
    bias=lora_cfg['bias'],
    task_type=lora_cfg['task_type'],
    inference_mode=False
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==================== 5. 数据预处理 (最关键部分) ====================
print("6. 加载并预处理数据集...")
# 1. 加载原始数据集
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

# 2. 定义格式化函数
def formatting_prompts_func(example):
    """将对话消息格式化为纯文本"""
    messages = example['messages']
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

# 3. 应用格式化并分词
def preprocess_and_tokenize(example):
    """预处理函数：生成文本并分词"""
    text = formatting_prompts_func(example)
    # 分词时不填充，由训练器内部的collator统一处理
    tokenized = tokenizer(text, truncation=True, max_length=2048)
    return tokenized

# 4. 创建最终的数据集
print("  正在预处理数据集...")
tokenized_dataset = dataset.map(
    preprocess_and_tokenize,
    remove_columns=dataset.column_names,  # 移除原始列，避免KeyError
    desc="预处理中",
)
print(f"  预处理完成。数据集示例键名: {list(tokenized_dataset[0].keys())}")

# ==================== 6. 配置训练参数 ====================
print("7. 配置训练参数...")
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    eval_strategy="no",
    report_to="none",
    remove_unused_columns=True,  # 设为True，因为我们已移除所有未用列
    dataloader_num_workers=4,
    max_seq_length=2048,
    dataset_text_field="input_ids",  # ✅ 关键修改：告诉训练器数据的主键
    packing=False,
)

# ==================== 7. 创建训练器并开始训练 ====================
print("8. 创建训练器...")
# ✅ 关键修改：传入预处理后的数据集
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=tokenized_dataset,  # ✅ 传入 tokenized_dataset，不是原始 dataset
    tokenizer=tokenizer,
    # 不再需要 formatting_func，因为数据已预处理
)

print("="*50)
print("开始训练！")
print("="*50)
trainer.train()

# ==================== 8. 保存训练结果 ====================
print("9. 保存模型与配置...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
with open(Path(OUTPUT_DIR) / "adapter_config.json", 'w') as f:
    json.dump(original_config['lora'], f, indent=2)
with open(Path(OUTPUT_DIR) / "chat_template.jinja", 'w') as f:
    f.write(original_config['chat_template'])

print(f"✅ 训练完成！LoRA适配器保存在: {OUTPUT_DIR}")