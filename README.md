# AI 学习助手 — 多模态 AI 应用 (Week 5)

能看、会听、能说的 AI 学习助手。拍照提问 + 语音交互 + AI 语音讲解，多模态融合在一个流畅的产品里。

## 核心能力

- **拍照提问**：上传或拍摄题目图片，图片直接进多模态模型（零信息损失），模型看图讲解
- **语音提问**：麦克风录音 → ASR 语音识别 → 文字进模型推理
- **AI 语音讲解**：模型回答 → TTS 合成语音 → 前端播放
- **多模态融合 + 上下文**：图片与后续语音/文字追问处在同一会话中，追问时模型仍"记得"那张图
- **流式输出**：文字回答 SSE 流式推送，实时显示推理过程
- **ReAct Agent**：自主编排工具调用链（知识库搜索、网络搜索、计算器、时间查询）

## 架构设计：模态原生输入

**主推理模型是视觉模型 (glm-4.6v)**，图片以结构化 content 直接进入对话：

```
图片 → [type: image_url] → glm-4.6v（看图 + Function Calling）→ 调用其他工具 → 推理 → 回答
```

**语音链路**：

```
麦克风录音 → MediaRecorder(webm) → ASR(Whisper/Zhipu) → 文字 → LLM推理 → TTS(CogTTS/tts-1) → Audio播放
```

**为什么这样设计：**

- **信息零损失**：图片作为结构化 content 直入主模型，模型能直接"看"到每个像素。避免了"图片→视觉模型→文字摘要→文本模型"的两次转译带来的信息丢失。
- **语音闭环**：完整的"听→想→说"链路，前端 MediaRecorder 录音 → ASR → LLM → TTS → Audio 播放。
- **一步直达**：同一个模型边看图边 function calling，不需要先调 analyze_image 拿摘要、再交给其他人继续推理。
- **延迟更低**：少一次 API 调用、少一次 token 中转。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（复制 .env.example 为 .env，填入 API Key）
cp .env.example .env

# 3. 启动服务
uvicorn main:app --reload --port 8000

# 4. 浏览器访问
http://localhost:8000
```

## API 端点

### Agent 端点 (`/api/v1/agent`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/agent/chat` | POST | Agent 对话（SSE 流式输出），支持 `image_url` 和 `image_base64` |
| `/api/v1/agent/sessions` | GET | 会话历史列表 |
| `/api/v1/agent/sessions/{id}/trace` | GET | 完整推理轨迹（JSON） |
| `/api/v1/agent/evaluate` | POST | 多模板对比评估 |

### 多模态端点 (`/api/v1/multimodal`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/multimodal/chat` | POST | 多模态对话（SSE 流式），支持文字 + 图片 + 语音混合输入 |
| `/api/v1/multimodal/voice` | POST | 语音问答闭环：音频 → ASR → LLM → TTS → 音频返回 |
| `/api/v1/multimodal/asr` | POST | 语音转文字（multipart 上传音频文件） |
| `/api/v1/multimodal/tts` | POST | 文字转语音（返回 audio/mpeg） |
| `/api/v1/multimodal/sessions` | GET | 多模态会话列表 |
| `/api/v1/multimodal/upload/image` | POST | 图片上传（返回 base64 data URI） |

### SSE 事件类型

多模态 `/chat` 端点返回以下 SSE 事件：

| 事件 | 说明 |
|------|------|
| `asr_result` | 语音识别结果 |
| `thought` | Agent 推理思考 |
| `action` | 工具调用动作 |
| `observation` | 工具执行结果 |
| `answer` | 最终回答 |
| `done` | 推理完成（含 session_id） |
| `session` | 多模态会话 ID |

### 使用示例

```bash
# 拍照提问
curl -X POST http://localhost:8000/api/v1/multimodal/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "这道题怎么解？", "image_base64": "data:image/png;base64,..."}'

# 语音提问（同一会话追问）
curl -X POST http://localhost:8000/api/v1/multimodal/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "第二步再详细讲一遍", "session_id": "abc123def456"}'

# 语音闭环
curl -X POST http://localhost:8000/api/v1/multimodal/voice \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "...", "language": "zh"}'
```

## 内置工具

| 工具 | 功能 |
|------|------|
| `search_knowledge_base` | 查询本地 ChromaDB 多模态 RAG 知识库 |
| `search_web` | 模拟联网搜索（Mock 数据） |
| `calculator` | 安全的数学表达式求值（AST 解析） |
| `get_current_time` | 获取当前日期时间 |

## 项目结构

```
├── app/
│   ├── routers/
│   │   ├── agent.py              # Agent API 路由
│   │   └── multimodal.py         # 多模态 API 路由（ASR/TTS/Chat/Voice）
│   ├── schemas/
│   │   ├── agent.py              # Agent Pydantic 数据模型
│   │   └── multimodal.py         # 多模态 Pydantic 数据模型
│   ├── services/
│   │   ├── agent.py              # ReAct Agent 引擎（多模态直达）
│   │   ├── multimodal_chat.py    # 多模态对话服务（会话管理 + 上下文）
│   │   ├── asr.py                # ASR 语音识别（Whisper + 智谱，带 fallback）
│   │   ├── tts.py                # TTS 语音合成（OpenAI + 智谱，带 fallback）
│   │   ├── tools.py              # 工具注册中心（4 工具）
│   │   ├── prompts.py            # 3 种 ReAct Prompt 模板
│   │   ├── llm.py                # LLM 调用封装
│   │   ├── evaluator.py          # LLM-as-Judge 评估器
│   │   ├── vector_store.py       # ChromaDB 向量存储
│   │   ├── retriever.py          # 多模态检索器
│   │   ├── embedding.py          # Embedding 服务
│   │   ├── clip_embedding.py     # CLIP 图片嵌入
│   │   └── sessions.py           # 会话管理（JSON 持久化）
│   └── utils/config.py           # 配置管理
├── static/index.html             # 多模态交互前端（拍照/录音/文字 + 语音播放）
├── tests/
│   ├── conftest.py               # 测试 fixtures（LLM/ASR/TTS mock）
│   ├── test_agent.py             # Agent 测试（12 个用例）
│   └── test_multimodal.py        # 多模态测试（17 个用例）
├── main.py                       # FastAPI 入口
└── requirements.txt
```

## 运行测试

```bash
pytest tests/ -v
# 46 passed
```

## 安全边界

- 图片上传限制：类型校验（png/jpeg/webp/gif）+ 大小限制（默认 10MB）
- 音频上传限制：类型校验（webm/wav/mp3 等）+ 大小限制（默认 10MB）+ 时长限制（默认 120s）
- 错误降级：ASR/TTS 双提供商 fallback，主备切换
- 输入校验：Pydantic 字段验证 + `model_validator` 确保至少一项输入

## 技术栈

- **后端**：FastAPI + SSE
- **Agent 引擎**：ReAct 模式 + OpenAI SDK Function Calling
- **主 LLM**：智谱 AI glm-4.6v（多模态视觉模型，106B参数，原生支持 Function Calling）
- **ASR**：OpenAI Whisper / 智谱 GLM-ASR
- **TTS**：OpenAI tts-1 / 智谱 CogTTS
- **向量库**：ChromaDB
- **前端**：原生 HTML/CSS/JS（MediaRecorder + SSE + Audio API）
- **测试**：pytest + httpx（46 个用例）
