# 基于 Nanobot 的多平台智能个人助理

<div align="center">

[![Forked from HKUDS/nanobot](https://img.shields.io/badge/Forked%20from-HKUDS%2Fnanobot-8A2BE2?logo=github)](https://github.com/HKUDS/nanobot)
[![Upstream License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
![Python](https://img.shields.io/badge/python-≥3.11-blue)
[![New Code](https://img.shields.io/badge/new%20code-8400%2B%20lines-ff6f61)](#-本仓库新增功能)
[![Tests](https://img.shields.io/badge/new%20tests-2400%2B%20lines-2ea44f)](#-工程质量)

</div>

**一个可 7×24 小时运行的个人 AI 助理**:接入飞书 / QQ / Telegram / Discord 等十余个聊天平台,具备 RAG 知识库、学术文献检索、PDF 智能解析、语音转写、图像与视频生成能力,并把模型的思考过程与工具调用实时展示给你。

> [!NOTE]
> 本仓库是 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) 的个人增强版,基于上游 2026-05-30 版本(约 v0.2.0)fork 后独立演进,聚焦「个人知识管理 + 中文平台体验 + 多模态交互」三个方向。上游在此之后仍在持续更新,两者互不影响。

## 📊 与上游仓库的关系

| 维度 | 说明 |
|------|------|
| 上游项目 | [HKUDS/nanobot](https://github.com/HKUDS/nanobot) — 轻量开源 AI Agent 框架(消息总线 + Agent Loop + 工具体系) |
| 本仓库定位 | 在保留原有轻量内核的基础上,按个人助理场景定制增强的**实际在用系统**(部署于 Azure,systemd 常驻运行) |
| 改动规模 | 20 个功能 commit · 54 个文件 · **+8402 / -154 行** · 新增 **12 个测试文件约 2400 行** |
| 演进方向 | 上游做通用框架;本仓库专注 RAG 知识库、中文平台深度适配、多模态生成与执行过程可视化 |

---

## 🎯 本仓库新增功能

### 1. RAG 知识库管道(全新模块 `nanobot/rag/`)

从零实现的端到端检索增强生成管道,约 1400 行:

- **本地向量存储**:基于 FAISS 的持久化向量库([vector_store.py](nanobot/rag/vector_store.py))
- **嵌入服务**:SiliconFlow BAAI/bge-m3(1024 维),OpenAI 兼容接口([embedding.py](nanobot/rag/embedding.py))
- **重排服务**:BAAI/bge-reranker-v2-m3 二阶段精排,提升召回质量([reranker.py](nanobot/rag/reranker.py))
- **标题感知分块**:按 Markdown 标题层级切分文档,保留语义完整性([chunker.py](nanobot/rag/chunker.py))
- **统一管道**:解析 → 分块 → 嵌入 → 检索 → 重排 → 上下文组装([pipeline.py](nanobot/rag/pipeline.py)),并以 `rag` 工具形式暴露给 LLM

### 2. 学术文献检索工具([literature.py](nanobot/agent/tools/literature.py))

- 聚合 **Semantic Scholar + arXiv** 双数据源
- 支持年份区间过滤、引用数、摘要、PDF 链接
- 与 RAG 联动:检索到的文献可直接入库沉淀为个人知识

### 3. PDF 智能解析与自动入库

- **MinerU 云端 API 解析**([mineru_parser.py](nanobot/utils/mineru_parser.py)):云端模式规避本地模型数 GB 内存开销,适配低配服务器
- **裸 PDF 自动流程**([loop.py](nanobot/agent/loop.py) `_ingest_bare_documents`):用户上传扫描件/无文本层 PDF 时自动触发「MinerU 解析 → 归档为 Markdown → RAG 入库 → 生成摘要回复」,全程无需手动操作
- 文档语言自动检测,不硬编码语言参数

### 4. 视频与图像生成

- **视频生成工具**([providers/video_generation.py](nanobot/providers/video_generation.py) + [tools/video_generation.py](nanobot/agent/tools/video_generation.py)):接入 OpenAI Videos 兼容 API,支持文本生成 / 关键帧 / 参考图三种模式;任务轮询上限 600s,队列满自动重试,本地素材 base64 内联上传
- **图像生成 API 路由**([api/server.py](nanobot/api/server.py)):新增 `/v1/images/generations` 与 `/v1/image/generate` OpenAI 兼容端点,程序化调用图像生成

### 5. QQ 频道深度增强([qq.py](nanobot/channels/qq.py),+415 行)

- **语音消息转写**:下载 `voice_wav_url` 后调用硅基流动 XingChenASR 转写,官方 `asr_refer_text` 兜底;修复了 botpy 附件解析丢弃 QQ 专有字段的问题
- **C2C 流式消息**:接入官方 `stream_messages` API(`input_mode=replace`),逐段推送打字机效果;采用 append-only 事件流设计规避 40007 错误
- **思考过程外显**:模型推理(引用块)+ 工具调用(🔧 块)按到达顺序实时渲染,与最终回答同卡片展示
- **Markdown 富文本输出** + 0.8s 节流;群聊无流式接口,自动降级为一次性发送

### 6. 飞书执行过程可视化([feishu.py](nanobot/channels/feishu.py),+522 行)

- 基于 CardKit 流式卡片的**可折叠「⚙️ 执行过程」面板**:模型的思考与工具调用全过程对用户透明
- **时间线交错展示**:思考与工具调用按真实发生顺序排列于单条时间线(而非分区割裂),多轮以 `···` 分隔,完成后自动收起
- 修复流式面板不实时刷新的问题(思考结束即推送完整内容,工具调用无缓冲时也立即建卡)
- 图片/文件上传失败时显式提示(含权限排查指引),不再静默丢弃

### 7. 会话记忆可靠性修复([session/manager.py](nanobot/session/manager.py))

修复三个上游记忆归档缺陷:`enforce_file_cap` 重复归档与消息丢失、非连续保留区间 `last_consolidated` 跟踪错误、空闲压缩丢弃内容未归档。

### 8. 其他增强

- **语音转写 Provider**([transcription.py](nanobot/providers/transcription.py)):新增 SiliconFlow 支持,模型/密钥/端点均可配置
- **resume-writer 技能**:内置 LaTeX 中文简历模板与写作规范的简历撰写技能
- **video-generation 技能**:视频生成的使用说明与最佳实践

---

## 💡 为什么基于 nanobot

- **持久工作流**:目标、记忆、工具与聊天上下文跨轮次存活,支持长任务
- **聊天原生触达**:WebUI、API、飞书、QQ、Telegram、Discord、Slack 等十余平台
- **模型自由**:任意 OpenAI 兼容 API、本地 LLM、图像生成、搜索与降级链
- **小内核**:可读性强的核心代码,MCP、记忆、部署、自动化开箱即用

## 📦 安装

**从源码安装(推荐,包含本仓库全部新功能)**

```bash
git clone https://github.com/Discard-001/nanobot-assistant.git
cd nanobot-assistant
pip install -e .
```

> [!TIP]
> 上游稳定版可从 PyPI 安装:`pip install nanobot-ai`(不含本仓库新增功能)

## 🚀 快速开始

**1. 初始化配置**

```bash
nanobot onboard
```

**2. 配置模型(`~/.nanobot/config.json`)**

```json
{
  "providers": {
    "openrouter": { "apiKey": "sk-or-v1-xxx" }
  },
  "agents": {
    "defaults": { "provider": "openrouter", "model": "anthropic/claude-opus-4-6" }
  }
}
```

**3. 启动**

```bash
nanobot gateway   # 常驻网关:WebUI + 各聊天频道
nanobot agent     # 或终端直接对话
```

**4.(可选)启用本仓库新增能力**

<details>
<summary>RAG 知识库 + MinerU PDF 解析</summary>

```json
{
  "tools": {
    "mineru": { "apiToken": "your-mineru-token" },
    "rag": {
      "embedding": {
        "provider": "siliconflow",
        "apiKey": "your-sf-key",
        "model": "BAAI/bge-m3"
      },
      "reranker": {
        "provider": "siliconflow",
        "apiKey": "your-sf-key",
        "model": "BAAI/bge-reranker-v2-m3"
      }
    }
  }
}
```

配置后:直接给机器人发 PDF,带文本层的正常阅读,扫描件自动走「MinerU 解析 → RAG 入库 → 摘要」流程。
</details>

<details>
<summary>QQ 频道(语音转写 + 流式 + Markdown)</summary>

```json
{
  "channels": {
    "qq": {
      "enabled": true,
      "streaming": true,
      "markdown": true
    }
  },
  "tools": {
    "transcription": {
      "provider": "siliconflow",
      "apiKey": "your-sf-key",
      "model": "FunAudioLLM/SenseVoiceSmall"
    }
  }
}
```
</details>

<details>
<summary>飞书 / 视频生成等更多配置</summary>

参见 [docs/configuration.md](./docs/configuration.md) 与 [docs/chat-apps.md](./docs/chat-apps.md)。视频/图像生成需在 `providers` 中配置对应网关的 API Key。
</details>

## 🏗️ 架构

<p align="center">
  <img src="images/nanobot_arch.png" alt="nanobot architecture" width="800">
</p>

消息经由异步 `MessageBus` 解耦:各频道(飞书/QQ/...)发布入站消息 → `AgentLoop` 构建上下文 → `AgentRunner` 驱动 LLM 多轮对话与工具执行 → 出站消息流回频道。本仓库新增的 `rag/`(知识库)、`literature`(文献)、`video_generation`(视频)以工具形式挂载进同一循环,飞书/QQ 的流式渲染在频道层实现,不侵入内核。

## 🖥️ WebUI

WebUI 随 wheel 内置,启用 WebSocket 频道后浏览器直接访问:

```json
{ "channels": { "websocket": { "enabled": true } } }
```

```bash
nanobot gateway   # 打开 http://127.0.0.1:8765
```

<p align="center">
  <img src="images/nanobot_webui.png" alt="nanobot webui preview" width="900">
</p>

## ✨ 场景演示

<table align="center">
  <tr align="center">
    <th>📈 实时信息分析</th>
    <th>🚀 全栈软件工程</th>
    <th>📅 智能日程管理</th>
    <th>📚 个人知识助理</th>
  </tr>
  <tr>
    <td align="center"><img src="case/search.gif" width="180" height="400"></td>
    <td align="center"><img src="case/code.gif" width="180" height="400"></td>
    <td align="center"><img src="case/schedule.gif" width="180" height="400"></td>
    <td align="center"><img src="case/memory.gif" width="180" height="400"></td>
  </tr>
  <tr>
    <td align="center">检索 · 洞察 · 趋势</td>
    <td align="center">开发 · 部署 · 扩展</td>
    <td align="center">日程 · 自动化 · 组织</td>
    <td align="center">学习 · 记忆 · 推理</td>
  </tr>
</table>

结合本仓库新增能力,还可以:

- 群里发一份 PDF 论文扫描件 → 自动解析入库 → 追问细节时基于全文回答
- 「帮我查一下 2024 年以来关于 commit 分解的论文」→ 文献检索 + 引用数排序
- 语音丢一句「明天提醒我……」→ 转写理解 → 建立定时任务
- 「把这张照片做成 8 秒视频」→ 关键帧模式视频生成

## 🧪 工程质量

- 每个新功能均配套测试:新增 **12 个测试文件、约 2400 行**(RAG 管道、裸 PDF 入库、MinerU 解析、QQ 流式/转写、飞书面板、视频生成、图像 API、会话归档等)
- 本地通过 `ruff` 检查与 `pytest` 全量验证后再部署
- 实际生产运行:Azure 服务器 systemd 常驻,`MemoryMax=300M` 硬约束下的内存优化实践(云端 MinerU、FAISS 轻量索引等选型均源于此约束)

## 📚 文档

- 聊天平台接入:[docs/chat-apps.md](./docs/chat-apps.md)
- 完整配置项:[docs/configuration.md](./docs/configuration.md)
- OpenAI 兼容 API / Python SDK:[docs/openai-api.md](./docs/openai-api.md) · [docs/python-sdk.md](./docs/python-sdk.md)
- Docker / Linux 服务部署:[docs/deployment.md](./docs/deployment.md)

## 🙏 致谢

本项目基于 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) 二次开发,感谢原作者 [Xubin Ren](https://github.com/re-bin) 与上游社区的开源贡献。上游的架构设计与轻量内核是本仓库所有扩展的基石,欢迎了解并支持原项目。

## ⭐ 上游 Star History

<div align="center">
  <a href="https://star-history.com/#HKUDS/nanobot&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/nanobot&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/nanobot&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/nanobot&type=Date" />
    </picture>
  </a>
</div>
