# 多轮对话意图捕捉系统

面向**保险智能营销与客服**场景的多轮对话意图识别系统。采用 **LLM 动态意图捕获 + 保险行业参考分类框架**，支持指代消解、意图漂移检测、结构化澄清引导与澄清回复 refinement。

## 核心特性

- **动态意图捕获**：DeepSeek / 阿里云千问 大模型理解用户真实诉求，输出自然语言意图描述，非固定枚举分类
- **保险行业参考框架**：12 类常见客户意图供 LLM 参考（产品咨询、保费询价、理赔服务等）
- **多轮上下文管理**：指代消解、槽位跨轮继承、主题栈追踪
- **意图澄清闭环**：模糊输入 → 结构化追问 → 用户回复 → 意图 refinement
- **意图漂移检测**：识别话题切换，区分相关意图链内切换
- **双入口**：交互式 CLI + FastAPI 生产接口

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 LLM API（复制并编辑）
cp .env.example .env
# DeepSeek: LLM_PROVIDER=deepseek + DEEPSEEK_API_KEY
# 千问:     LLM_PROVIDER=qwen + DASHSCOPE_API_KEY

# 交互式多轮对话
python chat.py

# 预设场景演示
python main.py --mode demo

# 自动化测试
python tests/run_tests.py

# 启动 API 服务
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## 项目结构

```
intention/
├── config/settings.py          # 系统配置（延迟预算、LLM、澄清阈值）
├── src/
│   ├── pipeline.py             # 主管道编排
│   ├── domain/                 # 保险领域参考框架
│   ├── engines/                # LLM 引擎 + 实体抽取
│   ├── context/                # 上下文管理与指代消解
│   ├── clarification/          # 澄清引导 + 澄清回复 refinement
│   ├── drift/                  # 意图漂移检测
│   └── models/                 # 数据模型
├── api/server.py               # FastAPI 接口
├── chat.py                     # 交互式对话入口
├── main.py                     # 演示入口
├── tests/                      # 自动化测试
└── docs/
    └── ARCHITECTURE.md         # 架构与技术文档（详细）
```

## 文档

| 文档 | 说明 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 整体架构、技术路线、实现思路 |
| [.env.example](.env.example) | 环境变量配置说明 |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供商：`deepseek` / `qwen` | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key（provider=deepseek） | — |
| `DEEPSEEK_MODEL` | DeepSeek 模型 | `deepseek-chat` |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope Key（provider=qwen） | — |
| `QWEN_MODEL` | 千问模型 | `qwen-plus` |
| `QWEN_API_BASE` | 千问 OpenAI 兼容端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_TIMEOUT_S` | 请求超时（秒） | `30` |
| `CLARIFICATION_CONFIDENCE_THRESHOLD` | 澄清触发置信度阈值 | `0.72` |

### 切换千问示例

```bash
# .env
LLM_PROVIDER=qwen
DASHSCOPE_API_KEY=sk-your-dashscope-key
QWEN_MODEL=qwen-plus

# 连通性测试
python scripts/test_llm.py
```

## API 示例

```bash
# 意图识别
curl -X POST http://localhost:8000/v1/intent/predict/sync \
  -H "Content-Type: application/json" \
  -d '{"utterance": "那它的等待期是多久？", "session_id": "user-001"}'

# 获取参考分类
curl http://localhost:8000/v1/intent/categories
```

## 技术指标（设计目标）

| 指标 | 目标 |
|------|------|
| 意图识别准确率 | ≥ 95% |
| 意图漂移检测率 | ≥ 92% |
| 多意图识别准确率 | ≥ 88% |
| 端到端延迟（含 LLM） | ≤ 600ms（轻量路径） / 1–3s（LLM 路径） |

## 技术栈

Python 3.10+ · Pydantic · httpx · FastAPI · DeepSeek / 阿里云千问 API（OpenAI 兼容）
