# 项目技术亮点清单（nanobot）

> 从 README / docs / 源码提炼，供简历编写时按需选取。每个亮点附带"简历话术方向"和"可量化角度"。

---

## 亮点 1：超轻量 Agent 核心循环与可观测执行链路

**技术要点**：
- 核心 AgentLoop 负责消息接入、上下文构建、模型调用、工具执行与回包
- 内置 Hook 体系支持进度回调、流式增量、工具事件起止通知
- 集成 AutoCompact 在长会话中自动压缩历史，控制上下文体积
- ToolRegistry 统一编排内置工具（文件、搜索、Shell、Web、Notebook 等）

**简历话术方向**：
- "设计超轻量 Agent 核心循环，统一消息接入、工具编排与流式响应，确保长会话稳定运行"
- "实现可观测执行链路与工具事件回调，支持逐步进度与流式输出同步到多渠道"

**可量化角度**：单次对话最大迭代数、工具调用次数、流式响应延迟、长会话压缩比例

---

## 亮点 2：多渠道聊天网关与插件化通道体系

**技术要点**：
- 统一 ChannelManager 路由 outbound 消息，支持多渠道并行启动
- 基于注册表扫描与 entry_points 机制加载频道插件
- 支持 Telegram / Discord / Slack / WeChat / WhatsApp / Feishu / DingTalk / Email / Matrix / QQ / WeCom / Teams 等
- Allow-list 访问控制与跨渠道进度提示

**简历话术方向**：
- "构建插件化聊天网关，统一管理 10+ 通信平台的会话与消息路由"
- "实现通道级权限控制与进度回传，保证跨平台交互一致性"

**可量化角度**：支持渠道数量、并发会话数、消息投递成功率、响应延迟

---

## 亮点 3：分层记忆系统（Consolidator + Dream + GitStore）

**技术要点**：
- 短期消息与长期记忆分层存储：session.messages / history.jsonl / SOUL.md / USER.md / MEMORY.md
- Consolidator 将旧对话压缩为结构化 history.jsonl 归档
- Dream 定期回读历史与长期文件，做最小化、可解释的记忆更新
- GitStore 记录长期记忆文件变更，支持审计与回滚

**简历话术方向**：
- "设计分层记忆体系，兼顾实时对话轻量性与长期知识沉淀"
- "引入 Dream 记忆更新与 GitStore 版本化，提升记忆可追溯与可恢复性"

**可量化角度**：历史压缩比例、Dream 处理批次大小、记忆文件变更次数、恢复耗时

---

## 亮点 4：MCP 工具接入与跨平台兼容

**技术要点**：
- 内置 MCP Client，将 MCP 服务器工具包装为原生工具
- 工具名称清洗与 JSON Schema 归一化，兼容 OpenAI/Anthropic Tool API
- 处理 transient 连接异常并自动重试，提高跨进程稳定性
- Windows 兼容 stdio 启动包装，提升本地工具可靠性

**简历话术方向**：
- "实现 MCP 工具接入层，统一工具名称与参数规范，兼容主流模型的工具调用协议"
- "补齐跨平台启动与断线重试能力，提升 MCP 工具稳定性"

**可量化角度**：接入 MCP 工具数量、工具调用成功率、重试后恢复比例

---

## 亮点 5：OpenAI 兼容 API 与多模态文件上传

**技术要点**：
- 提供 `/v1/chat/completions` 与 `/v1/models`，支持 OpenAI 兼容调用
- 支持 SSE streaming 与 session_id 会话隔离
- 支持 JSON base64 与 multipart 文件上传（图像与 Office/PDF 文档）
- API 会话与聊天渠道隔离，避免跨通道误发

**简历话术方向**：
- "实现 OpenAI 兼容 API，支持流式响应与会话隔离，便于本地集成"
- "支持多模态文件上传与解析，覆盖图片与常见文档格式"

**可量化角度**：文件上传大小上限、流式响应延迟、API 并发会话数

---

## 亮点 6：WebSocket 实时通道与多会话复用

**技术要点**：
- 内置 WebSocket Server，支持双向实时通信
- 支持流式 delta 与多 chat_id 复用的 multiplexing 协议
- Token 鉴权与 allowFrom 访问控制，内置死连接清理

**简历话术方向**：
- "实现 WebSocket 实时通道，支持流式增量输出与多会话复用"
- "设计 token 鉴权与 allow-list 访问控制，保障实时通道安全性"

**可量化角度**：并发连接数、消息延迟、流式分片大小、鉴权失败率

---

## 亮点 7：多 Provider 生态与配置化运维

**技术要点**：
- 支持 OpenAI / Anthropic / Bedrock / Groq / Gemini / DeepSeek / Qwen / Ollama 等多种 Provider
- 配置支持环境变量注入敏感信息，避免明文密钥
- 兼容各云厂商自定义 API Base 与兼容端点

**简历话术方向**：
- "构建多 Provider 适配层，支持云端与本地模型无缝切换"
- "通过配置化与环境变量注入实现安全、可迁移的部署方式"

**可量化角度**：支持 Provider 数量、切换零代码次数、配置项覆盖度

---

## 亮点 8：内置工具体系与安全执行边界

**技术要点**：
- 内置文件读写、搜索、Shell、Web、Notebook、Cron 等工具
- ToolRegistry 统一注册与路由，支持 progress/tool-hint 反馈
- 通过 workspace 约束与工具配置控制操作范围

**简历话术方向**：
- "构建内置工具体系与统一注册中心，提升 Agent 的可扩展与可控性"
- "通过工具配置与 workspace 约束设定安全边界，减少误操作风险"

**可量化角度**：内置工具数量、工具调用成功率、受限路径拦截次数
