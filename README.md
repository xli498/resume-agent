# Resume Agent

一个以真实性门禁为核心的简历与岗位匹配 Agent：读取候选人简历和岗位 JD，生成匹配分析、定向简历草稿、证据映射报告，并可选渲染带照片的简历图片。

> 当前定位：可审计的个人项目 / Demo。它不会替候选人核实事实，也不保证 ATS 评分、面试结果或录用结果。投递前必须由本人复核所有时间、数字、职责和成果。

## 特性

- 纯 Python 默认工作流；不依赖 LangGraph 即可运行
- 可选 `--workflow langgraph` 编排路径
- `sales` 商务销售稳重模板与 `ats` 单栏模板
- JD 动态关键词抽取、中英文词边界匹配和通用噪声过滤
- 复杂 JD 仅抽取可审计的已知能力概念与英文工具/证书名，不把标题、连接词或残句生成要求
- 岗位要求按硬性条件、优先条件和岗位职责分类，并映射到事实编号
- 对证据不足的要求生成待确认问题，确认前不会写入简历
- 逐条改写差异报告展示建议、证据及采纳或拒绝结果
- 禁止把 JD 要求、模型建议或推测直接写成候选人事实
- 数字、日期、强断言真实性门禁
- 最终简历逐条证据映射：直接命中、部分命中、需人工核对
- 原始简历事实账本（`F001...`）与模型改写 `evidence_ids` 强制绑定
- LangGraph 真实节点：事实提取、岗位分析、改写证据校验、定稿、真实性校验
- 图片动态高度、超长英文拆行、照片碰撞与边界检查
- 可同步生成 ATS Markdown、单页 A4 PDF、PNG 预览和 QA 报告
- 参考简历交付版式：白底单栏、右上竖版照片、模块标题下浅灰横线、宽边距
- 正文不低于约 10.5pt；一页放不下时明确要求精简低相关内容，不靠极小字号硬塞
- `--job-title` 可精确覆盖顶部“求职方向”；原始求职意向、工作职位或奖项中的真实“实习生”文本仍按事实保留
- 照片保持原比例，并自动裁掉证件照右侧近乎纯黑的附加画布
- 项目改写优先呈现技术/架构、核心实现、安全边界和验证结果；缺证据的维度自动省略
- 交付包写入版本化快照目录，并通过 `current-release.json` 单文件切换权威版本
- 每次运行清理本轮运行产物并生成 `run-manifest.json` 与外置校验文件 `run-manifest.sha256`
- LLM 接口错误脱敏；不输出响应正文、提示词或密钥
- 运行时不落盘完整提示词；调用日志仅记录请求状态，不记录模型名、端点或正文
- 输出目录拒绝符号链接，防止产物覆盖到非预期位置

## 快速开始

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 main.py --resume examples/resume.txt --jd examples/jd.txt --mock-llm
```

### 本地 MVP 工作台

不想记命令行参数时，可以启动本地 Web 工作台。它把一次岗位定向流程串起来：粘贴简历和 JD → 查看事实账本与待确认项 → 确认后生成可信定向简历 → 下载 Markdown。服务强制绑定回环地址，材料只在当前本机进程内处理；页面使用本地规则工作流，不会自动调用第三方模型。

```bash
python3 web_app.py
# 浏览器打开 http://127.0.0.1:8765
```

这是单用户本地 MVP，不是公网部署入口；程序会拒绝绑定 `0.0.0.0` 或其他非回环地址。任务状态暂存在内存中并受数量与有效期限制。PDF/PNG 交付包、DOCX/PDF 导入、真实模型确认流程和任务历史属于下一阶段。

默认输出到 `output/sales/`。不加 `--mock-llm` 或 `--call-llm` 时，程序只运行本地规则分析，不联网，也不会生成 `llm-analysis.json` 和定向草稿。需要完整演示输出时使用 `--mock-llm`；ATS 版本：

```bash
python3 main.py --resume examples/resume.txt --jd examples/jd.txt \
  --mock-llm --workflow langgraph --template ats --export-package \
  --job-title "大模型应用研发" --photo /path/to/photo.jpg
```

自定义输出目录：

```bash
python3 main.py --resume examples/resume.txt --jd examples/jd.txt \
  --mock-llm --output-dir /tmp/resume-agent-output
```

### 下载后使用自己的材料

1. 准备两个 UTF-8 文本文件：一份简历、一份岗位 JD。
2. 使用 `--resume` 和 `--jd` 指向它们；不要把真实材料提交到公开仓库。
3. 先用 `--mock-llm` 本地验证，再决定是否使用 `--call-llm`。

```bash
python3 main.py \
  --resume /path/to/my-resume.txt \
  --jd /path/to/job-description.txt \
  --mock-llm --template sales
```

程序默认只在本机读取文件；只有显式使用 `--call-llm` 时，才会把简历和 JD 发送到你配置的 OpenAI-compatible 接口。

生成图片时，使用你自己的照片路径；照片不会被提交到仓库：

```bash
python3 main.py --resume examples/resume.txt --jd examples/jd.txt \
  --mock-llm --render-image --photo /path/to/photo.jpg
```

## 真实模型调用（可选）

接口需要兼容 OpenAI 的 `/chat/completions`。只从环境变量读取：

```bash
export API_BASE_URL="https://your-endpoint/v1"
export API_KEY="your-key"
export MODEL="your-model"
python3 main.py --resume examples/resume.txt --jd examples/jd.txt --call-llm
```

不要把 API Key 写入代码、输入文件、日志或 Git。`.env.example` 仅是变量名示例，程序不会自动加载 `.env`。

LangGraph 是可选依赖：

```bash
pip install -e '.[langgraph]'
python3 main.py --workflow langgraph --resume examples/resume.txt --jd examples/jd.txt
```

未安装时，默认 Python 工作流仍可运行；显式选择 LangGraph 会明确报错，不会偷偷切换。

## 输出文件

每个模板使用独立目录：

- `final-resume.md`：最终简历
- `analysis.json`：结构化匹配结果
- `match-report.md`：岗位匹配报告
- `confirmation-questions.md`：当前缺少证据的岗位要求及确认问题
- `revision-diff-report.md`：模型改写的逐条证据与处理结果（仅模型模式）
- `evidence-mapping-report.md`：最终稿逐条证据映射
- `llm-analysis.json`：模型结构化结果（仅在 `--mock-llm` / `--call-llm` 时生成）
- `targeted-resume-draft.md`：定向候选草稿
- `final-resume-with-photo.png`：可选图片
- `run-manifest.json`：本轮产物清单
- `run-manifest.sha256`：运行清单的外置 SHA-256 校验值
- `final-resume-ats.md`：与最终稿同步的 ATS Markdown
- `final-resume.pdf`：单页 A4 PDF；默认正文不低于约 10.5pt，内容过多时明确失败并要求删减低相关内容
- `final-resume.png`：PDF 同源预览图
- `qa-report.json`：页数、文字提取、孤立标点、越界、分辨率和文件哈希检查
- `releases/<release_id>/`：包含 ATS/PDF/PNG 三件套和版本 manifest 的快照；同一运行账户仍可修改，完整性依赖 manifest 哈希复核
- `current-release.json`：当前权威版本指针；普通异常和已覆盖的进程中断会恢复旧指针

根目录的 ATS/PDF/PNG 是向后兼容副本，权威版本应从 `current-release.json` 解析。
`run-manifest.json`、`run-manifest.sha256` 和 `qa-report.json` 属于本次运行元数据，
不包含在三件套 release 快照内。目录与文件 `fsync` 可降低进程中断风险，但不能承诺
覆盖断电、内核崩溃或存储设备故障下的跨文件事务一致性。

模型返回的每条 `resume_revisions` 必须带 `evidence_ids`，并且只能引用
`analysis.json` 中事实账本的 `fact_id`。缺证据、引用无效证据、引入原文不存在的
数字或触发强断言门禁的改写不会进入最终简历。

## 测试

```bash
python3 -m py_compile main.py workflow.py llm_client.py test_resume_agent.py
python3 -m unittest -q
```

## 隐私与安全

仓库中的 `examples/` 只使用脱敏示例。真实简历、JD、照片、模型输出和日志均属于个人/运行时数据，不应提交到公开仓库。项目不上传输入文件；真实模型调用会把简历和 JD 发送到你配置的第三方模型接口，使用前请确认服务商的数据处理政策。

公开前请同时检查当前工作树、暂存区和 Git 历史；`.gitignore` 只能阻止未跟踪文件进入后续提交，不能从既有提交历史中删除已经提交过的内容。

## 许可证

MIT License，见 `LICENSE`。
