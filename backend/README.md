这个目前还实际上不是README 等我哪天改改）

## 后端：

本部分的目的是仿照`annali07`的`luna-sama`项目完成一组**LLM&TTS模型**的训练和部署。

### 数据获取：
通过工具`GARbro`解包游戏`月に寄りそう乙女の作法`，得到如下的文件目录：
```
__depack/
|   |-- avg/
|   |   |-- select/
|       |-- window/
|   |-- avg2/
|   |   |-- select/
|       |-- window/
|   |-- face/
|   |   |-- エッテ/
|   |   |-- サーシャ/
|   |   |-- スタンレー/
|   |   |-- その他/
|   |   |-- メリル/
|   |   |-- ユルシュール/
|   |   |-- ルナ/
|   |   |-- 北斗/
|   |   |-- 朝日/
|   |   |-- 大蔵りそな/
|   |   |-- 大蔵衣遠/
|   |   |-- 大蔵遊星/
|   |   |-- 紅葉/
|   |   |-- 花之宮瑞穂/
|   |   |-- 柳ヶ瀬湊/
|   |   |-- 名波七愛/
|   |   |-- 山吹八千代/
|       |-- 伊代/
|   |-- fade/
|       |-- アニメーション/
|   |-- include/
|   |-- include2/
|   |-- misc/
|   |-- scenario/
|   |   |-- アフター/
|   |   |-- エイプリル/
|       |-- 本編/
|   |-- script/
|   |   |-- albummode/
|   |   |-- aprillogo/
|   |   |-- apriltitle/
|   |   |-- config/
|   |   |-- dialog/
|   |   |-- dvdmediacheck/
|   |   |-- endroll/
|   |   |-- exceptionbackup/
|   |   |-- log/
|   |   |-- logo/
|   |   |-- memorymode/
|   |   |-- modecommon/
|   |   |-- musicmode/
|   |   |-- patchcheck/
|   |   |-- quit/
|   |   |-- reset/
|   |   |-- routeselect/
|   |   |-- saveload/
|   |   |-- sceneskipmenu/
|   |   |-- sceneskipmenu2/
|   |   |-- sequence/
|   |   |-- suspend/
|   |   |-- title/
|   |   |-- unlock/
|       |-- view/
|   |-- system/
|   |   |-- etc/
|   |   |-- font/
|       |-- voice/
|   |-- サムネイル/
|   |   |-- album/
|       |-- memory/
|   |-- ボイス/
|   |-- 画像/
|   |   |-- bg/
|   |   |-- cutin/
|   |   |-- effect/
|   |   |-- env/
|   |   |-- ev/
|   |   |-- ev_after/
|       |-- motion/
|   |-- 立ち絵/
|   |   |-- emotions/
|   |   |-- motion/
|   |   |-- エッテ/
|   |   |-- サーシャ/
|   |   |-- スタンレー/
|   |   |-- メリル/
|   |   |-- ユルシュール/
|   |   |-- ルナ/
|   |   |-- 北斗/
|   |   |-- 朝日/
|   |   |-- 大蔵りそな/
|   |   |-- 大蔵衣遠/
|   |   |-- 大蔵遊星/
|   |   |-- 紅葉/
|   |   |-- 花之宮瑞穂/
|   |   |-- 柳ヶ瀬湊/
|   |   |-- 名波七愛/
|   |   |-- 山吹八千代/
|       |-- 伊代/
|   |-- 効果音/
    |-- 音楽/
```


其中`scenario`中为所有的台本文件，`ボイス`中提供了所有的角色音频文件，而根据`face/角色名/`角色的表情，判断其情绪，再结合台本中的切换角色表情的指令，可以实现将这种指令文本替换为对应的情感标签，实现对台本的情感标记。

### 利用`ボイス`中的音频文件训练GPT-SoVITS模型

GPT-SoVITS是RVC-BOSS等作者研发的Text-To-Speech文字转语音的语音克隆合成模型，其完备的可视化训练流程对开发者非常友好，v2ProPlus(作者从v4技术路线回退至v2，其为v2的再优化版)表现已经相当出色，拟合语音迅速稳定。

在解包中，一共得到了6,763条角色`Asahi(朝日)`的音频文件，其依次通过GPT-SoVITS的语音切分、语音识别、训练集格式化（文本分词与特征提取、语音自监督特征提取和语音Token提取），经在一张3090上经过SoVITS微调训练、GPT微调训练后，得到ASA_e8_s6648.pth、ASA-e15.ckpt两模型的微调参数，将其替换入相应的权重位置，即可得到GPT-SoVITS模型。

### 利用`scenario`与情感Token微调Qwen3-8B模型（QLoRA）
在整合所有的.s台本后，可以得到`combined.txt`，形如：
```
％v_asa0001
【小倉朝日】
「大蔵家から紹介していただいた小倉と申します」
^chara03,file0:朝日/,file1:ASA_,file2:S_,file3:2_,file4:0_,file5:01

％v_yat0002
【インターホン】
「はい、話は伺っております。どうぞ、入り口までお越しください」

％v_yat0003
【山吹】
「はじめまして。小倉朝日様、本日はようこそおいでくださいました。当家のメイド長を務めます山吹と申します」
^chara01,file0:山吹八千代/,file1:YAT_,file2:S_,file3:0_,file4:0_,file5:04

％v_asa0002
【小倉朝日】
「あ、はじめまして」
^chara03,file5:06

％v_yat0004
【山吹】
「？」
^chara01,file5:01

％v_yat0005
【山吹】
「本日、小倉様は、当家のお雇いになるための面接に来た、と伺っておりますが」

％v_asa0003
【小倉朝日】
「あ、はいそう……です。メイド？　っていうのかな？　としてって聞いてます」
^chara03,file5:00

％v_yat0006
【山吹】
「雇用関係を結んでない以上、まだ小倉様はお客様に当たるのですが……もし本日から採用するとなれば、私はあなたの先輩になるのですよ。初対面から会釈での挨拶とは、これはまた大型新人のご登場ですね」
^chara01,file5:01

％v_asa0004
【小倉朝日】
「あっ！　し、失礼しました！」
^chara03,file5:15
```



根据立绘所得的情感匹配：
```
0	<E:smile>
1	<E:serious>
2	<E:thinking>
3	<M:eyes_closed>|<E:serious>
4	<E:worried>
5	<E:embarrassed>
6	<M:eyes_closed>|<E:smile>
7	<E:smile>
8	<M:eyes_closed>|<E:smirk>
9	<E:serious>
10	<E:angry>
11	<E:sad>
12	<E:sad>
13	<E:serious>
14	<E:resigned>
15	<E:shocked>
16	<E:shocked>
17	<E:surprised>
18	<E:smile>|<M:blush>
19	<E:serious>|<M:blush>
20	<E:thinking>|<M:blush>
```
整理格式，仅保留人名、台词和情感，得到`lines.txt`，形如：
```
小倉朝日】
「大蔵家から紹介していただいた小倉と申します」
<E:serious>
【インターホン】
「はい、話は伺っております。どうぞ、入り口までお越しください」
【山吹】
「はじめまして。小倉朝日様、本日はようこそおいでくださいました。当家のメイド長を務めます山吹と申します」
【小倉朝日】
「あ、はじめまして」
<M:eyes_closed>|<E:smile>
【山吹】
「？」
【山吹】
「本日、小倉様は、当家のお雇いになるための面接に来た、と伺っておりますが」
【小倉朝日】
「あ、はいそう……です。メイド？　っていうのかな？　としてって聞いてます」
<E:smile>
【山吹】
「雇用関係を結んでない以上、まだ小倉様はお客様に当たるのですが……もし本日から採用するとなれば、私はあなたの先輩になるのですよ。初対面から会釈での挨拶とは、これはまた大型新人のご登場ですね」
【小倉朝日】
「あっ！　し、失礼しました！」
<E:embarrassed>
```
最终，再按照储存历史等调整格式的方法，转换为实际的训练数据dataset.jsonl，形如：
```json
{"messages": [{"role": "system", "content": "あなたは【小倉朝日】として話してください。台詞は日本語で、原作の表記（「…」）を守ります。"}, {"role": "user", "content": "【前後の会話】\n【大蔵りそな】\n「今は側に居られればいいです。来年は同じ学校へ入学するつもりなので、仲良くしてください。妹、負けません」\n【大蔵りそな】\n「たかだか一日話しただけで男だとバレるようでは、三年間隠し通すなんてことはとても無理でしょう」\n【大蔵りそな】\n「そのためにフォロー役がいるなんて甘えは、とっ払った方がむしろあなたのためです。一人で行ってください」\n【インターホン】\n「どちら様でしょうか？」\n【小倉朝日】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）"}, {"role": "assistant", "content": "<E:serious>\n「大蔵家から紹介していただいた小倉と申します」"}], "chat_template_kwargs": {"enable_thinking": false}}
{"messages": [{"role": "system", "content": "あなたは【小倉朝日】として話してください。台詞は日本語で、原作の表記（「…」）を守ります。"}, {"role": "user", "content": "【前後の会話】\n【大蔵りそな】\n「そのためにフォロー役がいるなんて甘えは、とっ払った方がむしろあなたのためです。一人で行ってください」\n【インターホン】\n「どちら様でしょうか？」\n【インターホン】\n「はい、話は伺っております。どうぞ、入り口までお越しください」\n【山吹】\n「はじめまして。小倉朝日様、本日はようこそおいでくださいました。当家のメイド長を務めます山吹と申します」\n【小倉朝日】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）"}, {"role": "assistant", "content": "<M:eyes_closed>|<E:smile>\n「あ、はじめまして」"}], "chat_template_kwargs": {"enable_thinking": false}}
{"messages": [{"role": "system", "content": "あなたは【小倉朝日】として話してください。台詞は日本語で、原作の表記（「…」）を守ります。"}, {"role": "user", "content": "【前後の会話】\n【インターホン】\n「はい、話は伺っております。どうぞ、入り口までお越しください」\n【山吹】\n「はじめまして。小倉朝日様、本日はようこそおいでくださいました。当家のメイド長を務めます山吹と申します」\n【山吹】\n「？」\n【山吹】\n「本日、小倉様は、当家のお雇いになるための面接に来た、と伺っておりますが」\n【小倉朝日】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）"}, {"role": "assistant", "content": "<E:smile>\n「あ、はいそう……です。メイド？　っていうのかな？　としてって聞いてます」"}], "chat_template_kwargs": {"enable_thinking": false}}
{"messages": [{"role": "system", "content": "あなたは【小倉朝日】として話してください。台詞は日本語で、原作の表記（「…」）を守ります。"}, {"role": "user", "content": "【前後の会話】\n【山吹】\n「はじめまして。小倉朝日様、本日はようこそおいでくださいました。当家のメイド長を務めます山吹と申します」\n【山吹】\n「？」\n【山吹】\n「本日、小倉様は、当家のお雇いになるための面接に来た、と伺っておりますが」\n【山吹】\n「雇用関係を結んでない以上、まだ小倉様はお客様に当たるのですが……もし本日から採用するとなれば、私はあなたの先輩になるのですよ。初対面から会釈での挨拶とは、これはまた大型新人のご登場ですね」\n【小倉朝日】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）"}, {"role": "assistant", "content": "<E:embarrassed>\n「あっ！　し、失礼しました！」"}], "chat_template_kwargs": {"enable_thinking": false}}
```
最后再编写训练程序，在一张4090上完成训练，训练程序如下：
```python
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
    remove_unused_columns=True,  
    dataloader_num_workers=4,
    max_seq_length=2048,
    dataset_text_field="input_ids",  
    packing=False,
)

# ==================== 7. 创建训练器并开始训练 ====================
print("8. 创建训练器...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=tokenized_dataset,  # 传入 tokenized_dataset，不是原始 dataset
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

print(f"训练完成！LoRA适配器保存在: {OUTPUT_DIR}")
```
训练了3个epoch，训练过程如图，最终获得如下文件：
![alt text](3ecb696c085a61a20d8da7f5a81c6b0a.png)
![alt text](image.png)


## 两个模型的部署：

原作者在llm上采用流式输出，且仅支持一段文字输入，个人发现这样在反向传播（输出）速度和接口的灵活性上都比较差，故大幅更改了api.py:(不知道为啥这里有个超链接)

```python
#!/usr/bin/env python3
import sys
import torch
import argparse  # 新增：命令行参数解析
from typing import List, Dict, Optional, Any  # 新增Any类型
from fastapi import FastAPI
from pydantic import BaseModel, Field  # 新增Field
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import uvicorn

# ------------ 新增：命令行参数解析 ------------
parser = argparse.ArgumentParser(description='角色扮演LLM API服务')
parser.add_argument('--port', type=int, default=8000, 
                    help='服务端口号，默认8000')
args = parser.parse_args()

# ------------ Config ------------
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_DIR = "qwen3-asa-qlora"   # change to your adapter path
SYSTEM_PROMPT = "あなたは【小倉朝日】として話してください。台詞は日本語で、原作の表記（「…」）を守ります。"

# ------------ Model Load ------------
def ensure_chat_template(tok):
    if tok.chat_template and tok.chat_template.strip():
        return tok
    tok.chat_template = r"""
{% for message in messages %}
{% if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{% generation %}{{ message['content'] }}{% endgeneration %}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}
""".strip() + "\n"
    return tok

def load_model_and_tokenizer(base_id, adapter_dir, load_in_4bit=True):
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True, use_fast=False)
    tok = ensure_chat_template(tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            quantization_config=bnb,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok

def format_inputs(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt"
    )

print("Loading model…", file=sys.stderr)
model, tok = load_model_and_tokenizer(BASE_MODEL, ADAPTER_DIR, load_in_4bit=True)

# ------------ API Server ------------
app = FastAPI()

# ------------- 辅助函数 --------------
def format_conversation_history(history: List[Dict[str, str]], 
                               current_input: Optional[str] = None   ) -> str:
    
    formatted_lines = ["【前後の会話】"]
    
    # 处理历史消息
    for msg in history[-24:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if not content or not role:
            continue
        if role == "user":
            formatted_lines.append(f"【ユーザー】\n「{content}」")
        else:
            formatted_lines.append(f"【{role}】\n「{content}」")
    
    # 如果有用户输入
    if current_input and current_input.strip():
        formatted_lines.append(f"【ユーザー】\n「{current_input}」\n")

    formatted_lines.append(f"【小倉朝日】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）")
        
    return "\n".join(formatted_lines)

def parse_model_output(generated_text: str) -> tuple[str, str]:
    
    lines = [line.strip() for line in generated_text.strip().split('\n') if line.strip()]
    
    emotion, sentence = "", ""

    if len(lines) >= 1:
        emotion = lines[0]
        if not emotion.startswith('<E:') and not emotion.startswith('<M:'):
            emotion = ""
            sentence = generated_text.strip()
        elif len(lines) >= 2:
            sentence = lines[1]
    else:
        emotion = ""
        sentence = "「...」"
    
    if sentence and not sentence.startswith('「') and not sentence.startswith('『'):
        sentence = f"「{sentence}」"

    
    return emotion, sentence

# ------------ 服务信息模型 ------------
class ServiceInfo(BaseModel):
    """LLM服务信息"""
    character: str = Field(..., description="角色名称")
    port: int = Field(..., description="服务端口")
    host: str = Field(..., description="服务主机")
    model: str = Field(..., description="基础模型")
    adapter_dir: str = Field(..., description="适配器目录")
    # 新增：支持的历史格式信息
    supported_formats: Dict[str, Any] = Field(
        default_factory=lambda: {
            "history_format": "standard",
            "max_history_length": 24,
            "output_format": "emotion_tag + dialogue"
        },
        description="支持的格式信息"
    )

class ChatRequest(BaseModel):
    user: Optional[str] = Field( default_user="" , description="用户输入（可选）")  # 改为可选
    history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list,
        description="对话历史，格式：[{'role': 'user/assistant/角色名', 'content': '...'}]"
    )

class ChatResponse(BaseModel):
    emotion: str = Field(..., description="情感标签")
    sentence: str = Field(..., description="台词")

# ------------ 新增：/info 端点 ------------
@app.get("/info", response_model=ServiceInfo, summary="获取服务信息")
def get_service_info():

    return ServiceInfo(
        character="小倉朝日",
        port=args.port,
        host="0.0.0.0",
        model=BASE_MODEL,
        adapter_dir=ADAPTER_DIR,
        supported_formats={
            "history_format": "standard",
            "max_history_length": 24,
            "output_format": "emotion_tag + dialogue"
        }
    )

# ------------ 修改：/chat 端点 ------------
@app.post("/chat", response_model=ChatResponse, summary="对话接口")
def chat(req: ChatRequest):

    # 格式化输入：在LLM端进行格式化，确保与训练数据格式一致
    if req.history:
        # 有历史记录的情况
        formatted_input = format_conversation_history(
            history=req.history,
            current_input=req.user
        )
    else:
        # 无历史记录的情况（此时应有用户输入）
        formatted_input = f"【前後の会話】\n【ユーザー】\n「{req.user}」\n【小倉朝日】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_input}
    ]
    
    input_ids = format_inputs(tok, messages).to(model.device)
    
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=[tok.eos_token_id, eos_id],
        pad_token_id=tok.pad_token_id,
    )
    
    outputs = model.generate(input_ids, **gen_kwargs)
    
    # 解码输出，跳过输入部分
    input_length = input_ids.shape[1]
    generated_ids = outputs[0][input_length:]
    generated_text = tok.decode(generated_ids, skip_special_tokens=True)
    
    # 解析模型输出
    emotion, sentence = parse_model_output(generated_text)
    
    return ChatResponse(emotion=emotion, sentence=sentence)
    


# ------------ 新增：启动信息 ------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

同时，因为发现在本地WSL上同时运行两个4bit量化的llm模型对显存的要求远远超出一般笔记本的能力范围，考虑将模型在云端部署，因此大幅调整了gsv的api.py，使其能够返回可导出的语音文件：
```python
# gsv/api.py
import os, time, uuid, argparse, threading
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config_infer import Config
from .service import TTSService

def make_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="SoVITS API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    os.makedirs(args.out_dir, exist_ok=True)
    app.mount("/audio", StaticFiles(directory=args.out_dir), name="audio")

    is_half = (args.device.startswith("cuda") and not args.full_precision)
    sovits = args.sovits_path or args._cfg.pretrained_sovits_path
    gpt    = args.gpt_path    or args._cfg.pretrained_gpt_path
    svc = TTSService(args.device, is_half, args.hubert_path, args.bert_path, gpt, sovits)

    # Reference fixed at launch (can be changed later via /set_ref if desired)
    defaults = {
        "ref_wav":  args.ref_wav,
        "ref_text": args.ref_text,
        "ref_lang": args.ref_lang,
        "text_lang": args.text_lang or "zh",
        "basename": args.basename or "utt",
    }

    # Validate reference now so /speak never needs it
    if not os.path.exists(defaults["ref_wav"]):
        raise RuntimeError(f"ref_wav not found: {defaults['ref_wav']}")
    if not (defaults["ref_text"] and defaults["ref_lang"]):
        raise RuntimeError("ref_text/ref_lang must be provided at launch.")

    synth_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/config")
    def get_config():
        return {
            "device": args.device,
            "out_dir": os.path.abspath(args.out_dir),
            "defaults": defaults,
        }

    # Optional hot-swap of reference later
    @app.get("/set_ref")
    def set_ref(
        ref_wav: str = Query(None), ref_text: str = Query(None),
        ref_lang: str = Query(None), text_lang: str = Query(None),
        basename: str = Query(None),
    ):
        if ref_wav is not None:
            if not os.path.exists(ref_wav):
                raise HTTPException(400, f"not found: {ref_wav}")
            defaults["ref_wav"] = ref_wav
        if ref_text  is not None: defaults["ref_text"]  = ref_text
        if ref_lang  is not None: defaults["ref_lang"]  = ref_lang
        if text_lang is not None: defaults["text_lang"] = text_lang
        if basename  is not None: defaults["basename"]  = basename
        return {"ok": True, "defaults": defaults}

    # Inference: only text (+ optional text_lang)
    @app.get("/speak")
    def speak(
        text: str = Query(..., description="Target text to synthesize"),
        text_lang: str = Query(None, description="Override default text language"),
        speed: float = 1.0,
        top_k: int = 15, top_p: float = 0.6, temperature: float = 0.6,
        sample_steps: int = 32,
        basename: str = Query(None, description="Override output filename stem"),
    ):
        _text_lang = (text_lang or defaults["text_lang"]).strip()
        _ref_wav, _ref_text, _ref_lang = defaults["ref_wav"], defaults["ref_text"], defaults["ref_lang"]

        stem = (basename or defaults["basename"] or "utt")
        fname = f"{stem}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.wav"
        out_path = os.path.join(args.out_dir, fname)

        try:
            with synth_lock:
                sr, wav = svc.synth(
                    _ref_wav, _ref_text, _ref_lang, text, _text_lang,
                    top_k=top_k, top_p=top_p, temperature=temperature,
                    speed=speed, sample_steps=sample_steps,
                )
            sf.write(out_path, wav, sr)
        except Exception as e:
            raise HTTPException(500, f"synthesis failed: {e}")

        return {
            "ok": True,
            "sample_rate": sr,
            "url": f"/audio/{fname}",
            "path": os.path.abspath(out_path),
            "text_lang": _text_lang,
        }

    return app

def parse_args():
    g = Config()
    ap = argparse.ArgumentParser("gsv-api")

    # model/runtime
    ap.add_argument("-s","--sovits_path", default=g.sovits_path)
    ap.add_argument("-g","--gpt_path",    default=g.gpt_path)
    ap.add_argument("-hb","--hubert_path",default=g.cnhubert_path)
    ap.add_argument("-b","--bert_path",   default=g.bert_path)
    ap.add_argument("-d","--device",      default=g.infer_device)
    ap.add_argument("--fp","--full_precision", dest="full_precision", action="store_true")

    ap.add_argument("-dr","--default_ref_wav", default="")
    ap.add_argument("-dt","--default_ref_text", default="")
    ap.add_argument("-dl","--default_ref_lang", default="")

    # output
    ap.add_argument("--out_dir", default="out_api")
    ap.add_argument("--basename", default="utt")

    # reference (REQUIRED: pass via shell script at startup)
    ap.add_argument("--ref_wav",  required=True, help="Reference wav path")
    ap.add_argument("--ref_text", required=True, help="Reference transcript")
    ap.add_argument("--ref_lang", required=True, help="ja|zh|en|ko|yue or 中文/日文/…")

    # default text language for /speak
    ap.add_argument("--text_lang", default="zh")

    # server
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9881)

    args = ap.parse_args()
    args._cfg = g
    return args

def main():
    import uvicorn
    args = parse_args()
    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)

if __name__ == "__main__":
    main()
```



最终，两组模型分别在8000、8001、9880、9881运行，接口如下：
### 1）“桜小路ルナ”：（输入必须为日文）
**接口：117.50.85.179:8000（llm）9880（TTS）**
#### 8000：
支持的指令：
##### 1. /chat: 文字输入获得文字输出
###### 1.1 完整的用户输入和历史记录形式：
输入：`lty@DESKTOP-1VQFJJA:~$（这是控制台） curl -X POST http://117.50.85.179:8000/chat   -H "Content-Type: application/json"   -d '{    "user": "ルナ様もそう言っていました",      "history": [        {"role": "user", "content": "こんにちは"},       {"role": "桜小路ルナ", "content": "朝日、おはよう"},   {"role": "小倉朝日", "content": "<E:smile>\n「おはようございます、ルナ様」"}      ]    }'`
输出：`{"emotion":"<E:surprised>","sentence":"「…………」"}`
说明：此时为模型对用户的回应
###### 1.2 仅历史记录形式：
输入：`lty@DESKTOP-1VQFJJA:~$ curl -X POST http://117.50.85.179:8000/chat   -H "Content-Type: application/json"   -d '{    "user": "",      "history": [        {"role": "user", "content": "こんにちは"},       {"role": "桜小路ルナ", "content": "朝日、おはよう"},   {"role": "小倉朝日", "content": "<E:smile>\n「おはようございます、ルナ様」"}      ]    }'`
输出：`{"emotion":"<E:surprised>","sentence":"「あー……君は起きていたのか。私は寝てたんだが、いつからこ」"}`
说明：此时为模型对历史记录中最后一位说话者的回应
###### 1.3： 仅用户输入形式：
输入：`lty@DESKTOP-1VQFJJA:~$ curl -X POST http://117.50.85.179:8000/chat   -H "Content-Type: application/json"   -d '{    "user": "ルナ様もそう言っていました"   }'`
输出：`{"emotion":"<E:serious>","sentence":"「……ん？」"}`
说明：此时为模型对用户的回应
###### 1.4 总结：用户输入中的user字段内容可为空，history字段可以没有，输出为emotion标签+sentence文本形式
###### 1.5 处理要求；前端里储存的历史记录要翻译为日文，对于模型的输出只提取并保留其中的sentence内容，比如上面的…………；あー……君は起きていたのか。私は寝てたんだが、いつからこ；……ん？；
##### 2. /info：服务器信息
输入：`lty@DESKTOP-1VQFJJA:~$ curl  http://117.50.85.179:8000/info`
输出：`{"character":"桜小路ルナ","port":8000,"host":"0.0.0.0","model":"Qwen/Qwen3-8B","adapter_dir":"q
wen3-luna-qlora","supported_formats":{"history_format":"standard","max_history_length":8,"output_format":"emotion_tag + dialogue"} }`
处理：可以作为一个功能写在“模型”的功能最下面，也可以不处理

#### 9880：
对于每一个llm输出的文本回应，都应该将其发送给这个端口获取其对应的语音。比如llm输出了`遠慮なさらずおっしゃってください 私は絶対に笑ったりはしません`，则需要：
输入：`lty@DESKTOP-1VQFJJA:~$ curl -G "http://117.50.85.179:9880/speak"   --data-urlencode "text=遠慮なさらずおっしゃってください 私は絶対に笑ったりはしません"   --data-urlencode "text_lang=ja"`
输出：`{"ok":true,"sample_rate":32000,"url":"/audio/luna_1765817571155_294edb.wav","path":"/home/lty/luna-sama/out_repl/luna_1765817571155_294edb.wav","text_lang":"ja","download_info":{"download_id":"2d9285f5","download_url":"http://0.0.0.0:9880/download/2d9285f5","expires_at":"2025-12-15T18:52:56.271285","expires_in_minutes":120} }`
处理：此时应根据`http://0.0.0.0:9880/download/2d9285f5`，将其改为`http://117.50.85.179:9880/download/2d9285f5`，下载`luna_1765817571155_294edb.wav`并将其连接给对应的模型生成的文本。下载的文件应存储在一个固定的文件夹，比如`./out/repl/luna/luna_1765817571155_294edb.wav`,不同的模型存在不同的子文件夹，比如后面那个模型就可存储在./out/repl/asahi/
注意：应设置多次重复的（3-4次）下载请求直到文件下载成功或超时、被服务器拒绝，如果出现了下载失败的情况，不可使得程序崩溃，仅暂时关闭该文本对应的语音播放功能并提示一个错误发生了。

### 2）“小倉朝日”：（输入必须为日文）
接口：117.50.85.179:8001（llm）9881（TTS）
对于8001和9881的使用方法与1）完全相同，只需改变对应端口

