\# AI Gateway (Kelivo / OpenRouter 中转站 - 第一阶段)



\## 项目简介

这是一个最小可用的 FastAPI 网关服务，目标是先完成 \*\*OpenRouter API 中转\*\*，供 Kelivo 侧调用。

当前阶段已实现：



\- `GET /`：返回项目名 + 当前时间（UTC）

\- `GET /health`：健康检查

\- `POST /chat`：将用户消息转发到 OpenRouter Chat Completions

\- `/chat` 结构化日志：记录 `request\_id`、耗时、请求体、上游状态码



> 第一阶段坚持最小可用：不引入数据库、不引入复杂组件。



---



\## 依赖安装

> 推荐 Python 3.10+



\### Windows + PowerShell

```powershell

\# 1) 进入项目目录

cd .\\ai-gateway



\# 2) 创建虚拟环境

python -m venv .venv



\# 3) 激活虚拟环境

.\\.venv\\Scripts\\Activate.ps1



\# 4) 安装依赖

python -m pip install --upgrade pip

pip install -r requirements.txt

```



---



\## 配置说明

在项目根目录创建 `.env` 文件：



```env

OPENROUTER\_API\_KEY=你的\_openrouter\_api\_key

OPENROUTER\_MODEL=openai/gpt-4o-mini

```



\- `OPENROUTER\_API\_KEY`：必填，未配置时服务启动会报错。

\- `OPENROUTER\_MODEL`：可选，默认 `openai/gpt-4o-mini`。



---



\## 本地运行步骤

\### Windows + PowerShell

```powershell

\# 在激活虚拟环境后执行

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```



启动后访问：

\- `http://127.0.0.1:8000/`

\- `http://127.0.0.1:8000/health`

\- `http://127.0.0.1:8000/docs`



---



\## curl 测试命令

\### 1) 根路径版本信息

```bash

curl http://127.0.0.1:8000/

```



示例输出：

```json

{"project":"ai-gateway","time":"2026-02-25T12:34:56.789012+00:00"}

```



\### 2) 健康检查

```bash

curl http://127.0.0.1:8000/health

```



示例输出：

```json

{"ok":true}

```



\### 3) Chat 转发测试

```bash

curl -X POST "http://127.0.0.1:8000/chat" \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"message":"你好，做个自我介绍"}'

```



示例输出：

```json

{"reply":"你好！我是一个 AI 助手...","model":"openai/gpt-4o-mini"}

```



---



\## 运行日志示例（/chat）

服务每次处理 `/chat` 都会输出结构化日志，包含请求 ID、耗时、请求体、上游状态：



```text

2026-02-25 12:40:01,111 INFO chat\_request request\_id=6d2181f7-df9f-46c8-a2bb-8cad5a8f0770 duration\_ms=512.37 request\_body={'message': '你好，做个自我介绍'} upstream\_status=200

```





