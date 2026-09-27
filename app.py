
App · PY
# -*- coding: utf-8 -*-
"""瓦罗兰特 AI 复盘教练 · 后端
 
流水线（两段式）：
  1) 上传残局视频 -> ffmpeg 抽帧
  2) 帧序列 + 观察员 prompt -> Qwen3-VL（只做客观描述）
  3) 客观记录 + 结构化信息 + 教练 prompt -> 文本模型（指出不足）
  4) 返回 JSON 给前端渲染复盘卡
 
启动： uvicorn app:app --reload --port 8000
"""
 
import base64
import json
import os
import shutil
import tempfile
 
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
 
import prompts
from video import extract_frames
 
load_dotenv()
 
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
VL_MODEL = os.getenv("VL_MODEL", "qwen3-vl-plus")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen-plus")
FRAME_FPS = float(os.getenv("FRAME_FPS", "3"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "48"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
 
app = FastAPI(title="Valorant AI Coach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
def _client() -> OpenAI:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="未配置 DASHSCOPE_API_KEY。请复制 backend/.env.example 为 backend/.env 并填入你的百炼 API Key。",
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)
 
 
def _extract_json(text: str):
    """从模型输出里抽出第一个 JSON 对象/数组（容忍 ```json 代码块与前后废话）。"""
    text = (text or "").strip()
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        raise ValueError("模型输出中没有 JSON")
    end = None
    for j in range(len(text) - 1, -1, -1):
        if text[j] in "]}":
            end = j
            break
    if end is None or end <= start:
        raise ValueError("模型输出的 JSON 不完整")
    return json.loads(text[start:end + 1])
 
 
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "vl_model": VL_MODEL,
        "text_model": TEXT_MODEL,
        "api_key_set": bool(API_KEY),
    }
 
 
@app.post("/api/review")
async def review(
    video: UploadFile = File(...),
    map_name: str = Form(""),
    agent: str = Form(""),
    situation: str = Form(""),
):
    tmpdir = tempfile.mkdtemp(prefix="vcoach_")
    try:
        vpath = os.path.join(tmpdir, "clip.mp4")
        with open(vpath, "wb") as f:
            shutil.copyfileobj(video.file, f)
 
        # ---------- 抽帧 ----------
        frames = extract_frames(
            vpath, fps=FRAME_FPS, width=FRAME_WIDTH,
            max_frames=MAX_FRAMES, out_dir=tmpdir,
        )
        if not frames:
            raise HTTPException(status_code=400, detail="未能从视频中抽到画面，请确认文件是视频且 ffmpeg 已安装。")
 
        client = _client()
 
        # ---------- 第一段：Qwen3-VL 客观描述 ----------
        stamps = "、".join(t for t, _ in frames)
        user_text = (
            prompts.build_stage1_prompt(FRAME_FPS)
            + f"\n\n以下是按时间顺序抽取的 {len(frames)} 帧，时间戳依次为：{stamps}"
        )
        content = [{"type": "text", "text": user_text}]
        for _, path in frames:
            b64 = base64.b64encode(open(path, "rb").read()).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        r1 = client.chat.completions.create(
            model=VL_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
        )
        record = _extract_json(r1.choices[0].message.content)
 
        # ---------- 第二段：文本模型复盘 ----------
        prompt2 = prompts.build_review_prompt(
            map_name, agent, situation,
            json.dumps(record, ensure_ascii=False, indent=2),
        )
        r2 = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt2}],
            temperature=0.4,
        )
        review = _extract_json(r2.choices[0].message.content)
 
        return JSONResponse({"record": record, "review": review})
 
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"模型返回内容无法解析：{e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"复盘失败：{e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
 
 
if __name__ == "__main__":
    import uvicorn
 
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
 
