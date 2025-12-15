#!/usr/bin/env python3
import sys
import torch
import argparse  # 新增：命令行参数解析
from threading import Thread
from typing import List, Dict, Optional, Any  # 新增类型导入
from fastapi import FastAPI
from pydantic import BaseModel, Field  # 新增Field
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TextIteratorStreamer
)
from peft import PeftModel
import uvicorn

# ------------ 新增：命令行参数解析 ------------
parser = argparse.ArgumentParser(description='角色扮演LLM API服务')
parser.add_argument('--port', type=int, default=8000, 
                    help='服务端口号，默认8000')
parser.add_argument('--model-dir', type=str, default='qwen3-luna-qlora',
                    help='LoRA适配器目录，默认qwen3-luna-qlora')
args = parser.parse_args()

# ------------ Config ------------
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_DIR = args.model_dir   # 修改：使用命令行参数
SYSTEM_PROMPT = "あなたは【桜小路ルナ】として話してください。台詞は日本語で、原作の表記（「…」）を守ります。"

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

    formatted_lines.append(f"【桜小路ルナ】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）")
        
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

# ------------ 新增：服务信息模型 ------------
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
            "max_history_length": 8,
            "output_format": "emotion_tag + dialogue"
        },
        description="支持的格式信息"
    )

# ------------ 修改：请求模型 ------------
class ChatRequest(BaseModel):
    user: Optional[str] = Field( default_user="" , description="用户输入（可选）")  # 改为可选
    # 新增：历史记录字段
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
    """
    获取当前LLM服务的配置信息
    
    其他LLM或协调器可以通过此端点获取当前服务的设定信息
    """
    return ServiceInfo(
        character="桜小路ルナ",
        port=args.port,
        host="0.0.0.0",
        model=BASE_MODEL,
        adapter_dir=ADAPTER_DIR,
        supported_formats={
            "history_format": "standard",
            "max_history_length": 8,
            "output_format": "emotion_tag + dialogue"
        }
    )

# ------------ 修改：/chat 端点 ------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 修改：根据是否有历史记录格式化输入
    if req.history:
        # 有历史记录的情况
        formatted_input = format_conversation_history(
            history=req.history,
            current_input=req.user
        )
    else:
        # 无历史记录的情况（兼容简单模式）
        formatted_input = f"【前後の会話】\n【ユーザー】\n「{req.user}」\n【桜小路ルナ】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）"
    
    # 构建消息列表
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_input}
    ]
    
    input_ids = format_inputs(tok, messages).to(model.device)

    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=[tok.eos_token_id, eos_id],
        pad_token_id=tok.pad_token_id,
    )

    # 保持原作者的流式生成方式
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, skip_prompt=True)
    th = Thread(target=model.generate, kwargs={
        "inputs": input_ids,
        "streamer": streamer,
        **{k:v for k,v in gen_kwargs.items() if v is not None}
    })
    th.start()

    buf = []
    newline_count = 0
    for piece in streamer:
        # count newlines in this piece and keep only up to the 2nd newline
        if newline_count < 2 and "\n" in piece:
            parts = piece.split("\n")
            for i, part in enumerate(parts):
                if i < len(parts) - 1:            # this sub-part ends with a newline
                    buf.append(part + "\n")
                    newline_count += 1
                    if newline_count >= 2:
                        break
                else:
                    if newline_count < 2:
                        buf.append(part)          # last fragment (no newline)
            if newline_count >= 2:
                break
        else:
            buf.append(piece)

    # drain so the background thread can finish cleanly
    for _ in streamer:
        pass

    text = "".join(buf).rstrip() + "\n"           # ensure final newline

    # 修改：使用新的解析函数
    emotion, sentence = parse_model_output(text)
    
    return ChatResponse(emotion=emotion, sentence=sentence)

# ------------ 修改：启动部分 ------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)