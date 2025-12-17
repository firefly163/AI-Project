# gsv/api.py
import os, time, uuid, argparse, threading
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse  # 【新增导入】
from datetime import datetime, timedelta  # 【新增导入】
import json  # 【新增导入】

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
    
    # ====================== 【方案一：动态文件下载系统】 ======================
    # 存储最近生成的文件信息：{file_id: {"path": file_path, "expires_at": timestamp}}
    recent_files = {}
    recent_files_lock = threading.Lock()  # 用于线程安全的锁
    
    def cleanup_expired_files():
        """清理过期的文件记录"""
        with recent_files_lock:
            current_time = time.time()
            expired_keys = []
            for file_id, file_info in recent_files.items():
                if current_time > file_info["expires_at"]:
                    expired_keys.append(file_id)
            for key in expired_keys:
                del recent_files[key]
            if expired_keys:
                print(f"清理了 {len(expired_keys)} 个过期文件记录")
    # ====================================================================

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
    
    # ====================== 【新增：动态文件下载端点】 ======================
    @app.get("/download/{file_id}")
    async def download_file(file_id: str):
        """
        通过文件ID动态下载文件
        使用示例：http://117.50.85.179:9881/download/abc123
        """
        # 先清理过期文件
        cleanup_expired_files()
        
        if file_id not in recent_files:
            raise HTTPException(status_code=404, detail="文件不存在或已过期")
        
        file_info = recent_files[file_id]
        file_path = file_info["path"]
        
        if not os.path.exists(file_path):
            # 文件不存在，从记录中删除
            with recent_files_lock:
                recent_files.pop(file_id, None)
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 返回文件下载
        return FileResponse(
            file_path,
            filename=os.path.basename(file_path),
            media_type='audio/wav',
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"
            }
        )
    
    @app.post("/register_file")
    async def register_file(file_path: str, expires_minutes: int = 60):
        """
        注册新生成的文件，使其可通过动态链接下载
        使用示例：POST /register_file?file_path=/path/to/file.wav&expires_minutes=120
        """
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="文件路径不存在")
        
        # 生成唯一的文件ID（使用短UUID）
        file_id = str(uuid.uuid4())[:8]
        expires_at = time.time() + (expires_minutes * 60)
        
        with recent_files_lock:
            recent_files[file_id] = {
                "path": file_path,
                "expires_at": expires_at,
                "created": datetime.now().isoformat(),
                "size": os.path.getsize(file_path)
            }
        
        # 清理过期文件
        cleanup_expired_files()
        
        return {
            "file_id": file_id,
            "download_url": f"http://{args.host if args.host == '0.0.0.0' else '127.0.0.1'}:{args.port}/download/{file_id}",
            "public_url": f"http://{args.host if args.host == '0.0.0.0' else '127.0.0.1'}:{args.port}/audio/{os.path.basename(file_path)}",
            "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
            "expires_in_minutes": expires_minutes,
            "file_size": os.path.getsize(file_path),
            "filename": os.path.basename(file_path)
        }
    
    @app.get("/list_files")
    async def list_files(include_expired: bool = False):
        """
        列出所有已注册的文件（用于调试）
        """
        current_time = time.time()
        file_list = []
        
        with recent_files_lock:
            for file_id, info in recent_files.items():
                is_expired = current_time > info["expires_at"]
                if not include_expired and is_expired:
                    continue
                    
                file_list.append({
                    "file_id": file_id,
                    "filename": os.path.basename(info["path"]),
                    "path": info["path"],
                    "size": info["size"],
                    "created": info["created"],
                    "expires_at": datetime.fromtimestamp(info["expires_at"]).isoformat(),
                    "expired": is_expired,
                    "download_url": f"http://{args.host if args.host == '0.0.0.0' else '127.0.0.1'}:{args.port}/download/{file_id}"
                })
        
        return {
            "count": len(file_list),
            "files": file_list
        }
    # ====================================================================

    # Inference: only text (+ optional text_lang)
    @app.get("/speak")
    def speak(
        text: str = Query(..., description="Target text to synthesize"),
        text_lang: str = Query(None, description="Override default text language"),
        speed: float = 1.0,
        top_k: int = 15, top_p: float = 0.6, temperature: float = 0.6,
        sample_steps: int = 32,
        basename: str = Query(None, description="Override output filename stem"),
        auto_register: bool = Query(True, description="是否自动注册为可下载文件"),  # 【新增参数】
        expires_minutes: int = Query(120, description="下载链接有效期（分钟）")  # 【新增参数】
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
        
        # ====================== 【新增：自动注册生成的文件】 ======================
        download_info = None
        if auto_register:
            try:
                # 注册文件到动态下载系统
                file_id = str(uuid.uuid4())[:8]
                expires_at = time.time() + (expires_minutes * 60)
                
                with recent_files_lock:
                    recent_files[file_id] = {
                        "path": out_path,
                        "expires_at": expires_at,
                        "created": datetime.now().isoformat(),
                        "size": os.path.getsize(out_path)
                    }
                
                download_info = {
                    "download_id": file_id,
                    "download_url": f"http://{args.host if args.host == '0.0.0.0' else '127.0.0.1'}:{args.port}/download/{file_id}",
                    "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
                    "expires_in_minutes": expires_minutes
                }
            except Exception as e:
                print(f"文件注册失败（不影响正常生成）: {e}")
        # ====================================================================

        return {
            "ok": True,
            "sample_rate": sr,
            "url": f"/audio/{fname}",  # 原始静态文件URL
            "path": os.path.abspath(out_path),
            "text_lang": _text_lang,
            "download_info": download_info,  # 【新增：动态下载信息】
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
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9880)

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