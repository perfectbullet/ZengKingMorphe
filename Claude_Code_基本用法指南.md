# Claude Code 基本用法指南

## 简介

Claude Code 是 Anthropic 官方的 CLI 工具，用于在命令行环境中与 Claude AI 交互，辅助软件开发任务。

## Windows 系统特别说明

### 推荐的命令行工具

在 Windows 上，有以下选择：

1. **VS Code 集成终端**（推荐）⭐
   - 直接在 VS Code 中使用
   - 支持 PowerShell、CMD、Git Bash
   - 体验最佳，无缝集成

2. **Windows Terminal**（推荐）⭐
   - 微软官方终端
   - 支持 PowerShell、WSL、Git Bash 多标签页
   - 从 Microsoft Store 下载

3. **PowerShell**
   - Windows 内置，默认可用
   - 部分命令语法需要注意

4. **Git Bash**
   - Git for Windows 附带
   - 支持 Unix 风格命令
   - 适合习惯 Linux 的用户

5. **WSL (Windows Subsystem for Linux)**（强烈推荐用于复杂项目）⭐⭐⭐
   - 完整 Linux 环境，原生支持所有 Unix 命令
   - 适合需要原生 Linux 工具的场景
   - 与 Windows 文件系统无缝集成
   - 完美兼容 Python 虚拟环境、Node.js、Docker 等

### Windows 特定注意事项

✅ **完全兼容的功能**：
- 所有斜杠命令 (`/help`, `/clear`, `/exit` 等)
- 文件读取、编辑、创建
- 代码搜索（Glob/Grep）
- Git 操作
- VS Code 扩展功能

⚠️ **需要注意的功能**：
- Shell 命令：部分 Unix 命令在 Windows 上不可用
- 环境变量：设置方式不同
- 路径分隔符：使用 `\` 而非 `/`
- 虚拟环境激活：需要使用 `.ps1` 或 `.bat` 脚本

## 安装与启动

### 安装
```powershell
# 使用 npm（需要先安装 Node.js）
npm install -g @anthropic-ai/claude-code

# 验证安装
claude --version
```

### 启动

**方法 1：在 VS Code 中使用**（推荐）
```powershell
# 打开 VS Code 集成终端（Ctrl + `）
# 在项目目录下
cd D:\zenking_work\metahuman_work\ZengKingMorphe
claude
```

**方法 2：在 Windows Terminal 中**
```powershell
# 打开 Windows Terminal
# 切换到项目目录
cd D:\zenking_work\metahuman_work\ZengKingMorphe
claude
```

**方法 3：使用 VS Code 扩展**（最简单）
1. 安装 "Claude Code" VS Code 扩展
2. 打开命令面板（Ctrl + Shift + P）
3. 输入 "Claude Code: Start"
4. 直接在编辑器中使用

**方法 4：在 WSL 中使用**（推荐用于复杂项目）⭐
```bash
# 在 Windows Terminal 中打开 WSL
wsl

# 或者直接打开特定发行版
wsl -d Ubuntu-22.04

# 切换到项目目录（WSL 会自动挂载 Windows 驱动器）
cd /mnt/d/zenking_work/metahuman_work/ZengKingMorphe

# 启动 Claude Code
claude
```

## 基本命令

### 斜杠命令

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助信息 |
| `/clear` | 清空当前对话历史 |
| `/exit` | 退出 Claude Code |
| `/tasks` | 查看运行中的任务 |

### 内置快捷操作

| 命令 | 功能 |
|------|------|
| `/commit` | 创建 Git 提交（自动分析变更） |
| `/review-pr` | 审查 Pull Request |

## 核心功能

### 1. 代码操作
```bash
# 读取文件
"请读取 package.json"

# 编辑文件
"将 logger.debug 改为 logger.info"

# 创建新文件
"创建一个 utils/logger.py 文件"
```

### 2. 代码搜索
```bash
# 按文件名查找
"找到所有 .test.js 文件"

# 按内容搜索
"搜索所有包含 'axios' 的文件"

# 正则表达式搜索
"匹配所有的 console.log 语句"
```

### 3. Shell 命令执行
```bash
# 运行测试
"运行 npm test"

# 安装依赖
"安装 lodash 和 axios"

# Git 操作
"查看当前 git 状态"
```

### 4. 代码解释
```bash
# 解释代码功能
"解释 src/utils.js 中的 parseData 函数"

# 分析问题
"为什么这个 API 调用会失败？"
```

## 高级功能

### 计划模式 (Plan Mode)

对于复杂任务，Claude Code 会进入计划模式：

```bash
# 示例触发场景
"添加用户认证功能"
"重构数据库层"
"优化性能问题"
```

**计划模式流程**：
1. 探索代码库结构
2. 分析现有实现
3. 制定实施计划
4. 等待用户确认
5. 执行实施

### 并行工具调用

Claude Code 会自动并行执行独立操作：

```bash
# 同时读取多个文件
"读取 package.json, tsconfig.json 和 README.md"

# 并行运行独立测试
"运行所有单元测试和集成测试"
```

### 后台任务

```bash
# 长时间运行的任务
"在后台运行 docker-compose up"
"启动开发服务器"
```

查看后台任务：
```bash
/tasks
```

## 最佳实践

### 1. 提供清晰的上下文
```bash
# ✅ 好的做法
"在 app/services/user.py 中，修改 UserService 类的 login 方法，添加 JWT token 生成"

# ❌ 避免模糊指令
"修改登录功能"
```

### 2. 利用 IDE 集成

**VS Code 扩展**：
- 在编辑器中选择代码，右键可直接发送给 Claude
- 实时查看 Claude 的建议和修改
- 内联显示代码差异

### 3. 分阶段处理复杂任务

```bash
# 第一阶段：理解代码
"分析当前的认证流程"

# 第二阶段：制定计划
"如何添加 OAuth2 支持？"

# 第三阶段：实施
"按照刚才的计划实施 OAuth2"
```

### 4. 使用项目特定指令

在项目根目录创建 `.github/copilot-instructions.md` 或 `CLAUDE.md`：
```markdown
# 项目约定

- 使用 TypeScript strict 模式
- 所有 API 调用需要错误处理
- 遵循 RESTful 设计原则
```

## 常见使用场景

### 场景 1：Bug 修复
```bash
"修复登录接口的 500 错误"
```
Claude 会：
1. 读取相关代码
2. 分析错误日志
3. 定位问题
4. 修复代码
5. 运行测试验证

### 场景 2：功能开发
```bash
"添加用户头像上传功能"
```
Claude 会：
1. 理解需求
2. 设计 API 接口
3. 实现前后端代码
4. 添加数据验证
5. 编写测试

### 场景 3：代码重构
```bash
"将 UserService 拆分为多个小模块"
```
Claude 会：
1. 分析现有代码结构
2. 设计新的模块划分
3. 逐步迁移功能
4. 更新所有引用

### 场景 4：文档生成
```bash
"为 API 模块生成 JSDoc 注释"
```

### 场景 5：代码审查
```bash
"审查 src/api/ 目录下的所有文件"
```

## 配置选项

### Windows 环境变量设置

**方法 1：临时设置（当前会话）**
```powershell
# PowerShell
$env:ANTHROPIC_API_KEY="your-api-key"
$env:CLAUDE_MODEL="claude-sonnet-4.5"
$env:CLAUDE_MAX_CONCURRENT_TASKS="5"

# CMD
set ANTHROPIC_API_KEY=your-api-key
set CLAUDE_MODEL=claude-sonnet-4.5
```

**方法 2：永久设置（系统环境变量）**
```powershell
# 使用 setx（需要管理员权限）
setx ANTHROPIC_API_KEY "your-api-key" /M
setx CLAUDE_MODEL "claude-sonnet-4.5" /M

# 或者通过系统设置
# 控制面板 → 系统 → 高级系统设置 → 环境变量
```

**方法 3：用户级配置文件**
```powershell
# 在 PowerShell 配置文件中添加（推荐）
notepad $PROFILE

# 添加以下内容：
$env:ANTHROPIC_API_KEY="your-api-key"
$env:CLAUDE_MODEL="claude-sonnet-4.5"
```

### 配置文件

**Windows 配置文件位置**：
```powershell
# 用户配置路径
C:\Users\<你的用户名>\.claude\config.json

# 或使用环境变量
echo $env:USERPROFILE
```

在 `~/.claude/config.json` 中：
```json
{
  "model": "claude-sonnet-4.5",
  "maxConcurrentTasks": 5,
  "enableAutoCommit": false,
  "hooks": {
    "pre-edit": "npm run lint",
    "post-edit": "npm run format"
  }
}
```

**创建配置文件示例**：
```powershell
# 创建配置目录
mkdir $env:USERPROFILE\.claude

# 创建配置文件
notepad $env:USERPROFILE\.claude\config.json
```

## Hooks 自动化

配置自动化工作流：

```json
{
  "hooks": {
    "pre-commit": "npm run test && npm run lint",
    "user-prompt-submit": "echo '用户提交了新的提示'"
  }
}
```

## MCP 服务器集成

Claude Code 支持 Model Context Protocol (MCP) 扩展功能：

```bash
# 安装 MCP 服务器
claude mcp install github://anthropics/mcp-server-git

# 使用 MCP 功能
"获取最近 10 条 Git 提交记录"
```

## 故障排除

### Windows 常见问题与解决方案

**问题 1**: PowerShell 执行策略限制
```powershell
# 错误：无法加载文件，因为在此系统上禁止运行脚本
# 解决方案：以管理员身份运行 PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 或者临时绕过
powershell -ExecutionPolicy Bypass -File script.ps1
```

**问题 2**: Python 虚拟环境激活
```powershell
# 方法 1：PowerShell
D:\zenking_work\metahuman_work\ZengKingMorphe\.venv\Scripts\Activate.ps1

# 方法 2：CMD
D:\zenking_work\metahuman_work\ZengKingMorphe\.venv\Scripts\activate.bat

# 方法 3：Git Bash
source D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/activate

# 如果执行策略阻止，使用完整路径调用 Python
D:\zenking_work\metahuman_work\ZengKingMorphe\.venv\Scripts\python.exe script.py

# WSL 环境（推荐）使用以下路径
/mnt/d/zenking_work/metahuman_work/ZengKingMorphe/.venv/bin/python script.py
```

**问题 3**: 路径中的空格和特殊字符
```powershell
# ✅ 正确：路径用引号括起来
cd "D:\zenking work\metahuman work\ZengKingMorphe"

# ❌ 错误：不用引号会报错
cd D:\zenking work\metahuman work\ZengKingMorphe

# 或者使用 Tab 补全（自动处理引号）
cd D:\zen<Tab>  # 自动补全为正确格式
```

**问题 4**: Docker 在 Windows 上的路径
```powershell
# Docker 容器内使用 Unix 风格路径
# Windows 路径会自动转换
docker-compose up -d

# 查看日志
docker-compose logs -f ai-service

# 进入容器
docker exec -it <container_id> bash
```

**问题 5**: Git Bash 和 Windows 路径混合
```bash
# Git Bash 中可以混合使用
# Windows 风格：D:\path\to\file
# Unix 风格：/d/path/to/file
# 两者都可以正常工作
cd /d/zenking_work/metahuman_work/ZengKingMorphe
cd D:\zenking_work\metahuman_work\ZengKingMorphe
```

**问题 6**: 中文文件名或路径
```powershell
# 确保终端使用 UTF-8 编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:LANG = "zh_CN.UTF-8"

# VS Code 终端默认支持 UTF-8，无需额外配置
```

**问题 7**: Node.js 或 npm 命令找不到
```powershell
# 检查 Node.js 是否安装
node --version
npm --version

# 如果找不到，重新安装或添加到 PATH
# 从 https://nodejs.org 下载 LTS 版本
# 或使用 Windows 包管理器
winget install OpenJS.NodeJS.LTS
```

**问题 8**: Claude Code 命令不起作用
```powershell
# 检查 Claude Code 是否正确安装
claude --version

# 重新安装
npm uninstall -g @anthropic-ai/claude-code
npm install -g @anthropic-ai/claude-code

# 清除 npm 缓存
npm cache clean --force
```

## 获取帮助

- 官方文档: https://github.com/anthropics/claude-code
- 问题反馈: https://github.com/anthropics/claude-code/issues
- 使用 `/help` 查看内置帮助

## 本项目的特殊用法（Windows 环境）

### 启动开发环境

**方法 1：使用 Docker Desktop（推荐）**
```powershell
# 启动数据库服务
docker-compose up -d mongodb elasticsearch chroma

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

**方法 2：本地运行 AI 服务**
```powershell
# 1. 先启动数据库
docker-compose up -d mongodb elasticsearch chroma

# 2. 进入 ai-service 目录
cd ai-service

# 3. 激活虚拟环境
# PowerShell:
D:\zenking_work\metahuman_work\ZengKingMorphe\.venv\Scripts\Activate.ps1

# CMD:
D:\zenking_work\metahuman_work\ZengKingMorphe\.venv\Scripts\activate.bat

# 4. 启动服务
uvicorn main:app --reload --port 8000

# 5. 访问 API 文档
# 浏览器打开: http://localhost:8000/docs
```

**方法 3：完整 Docker 部署**
```powershell
# 启动所有服务（包括 AI 服务）
docker-compose up -d

# 查看所有日志
docker-compose logs -f

# 停止所有服务
docker-compose down
```

### 运行测试

**在 VS Code 终端中：**
```powershell
# 确保在项目根目录
cd D:\zenking_work\metahuman_work\ZengKingMorphe

# 运行所有测试
.venv\Scripts\python.exe -m pytest tests/

# 运行特定测试文件
.venv\Scripts\python.exe -m pytest tests/test_web_search.py

# 运行测试并显示输出
.venv\Scripts\python.exe -m pytest tests/ -v -s

# 运行测试并生成覆盖率报告
.venv\Scripts\python.exe -m pytest tests/ --cov=app --cov-report=html
```

**使用 pytest 配置文件：**
```powershell
# 如果项目有 pytest.ini 或 pyproject.toml，可以直接运行
.venv\Scripts\python.exe -m pytest
```

### 调试技巧

**VS Code 调试配置（.vscode/launch.json）：**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "main:app",
        "--reload",
        "--port",
        "8000"
      ],
      "cwd": "${workspaceFolder}/ai-service",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/ai-service"
      }
    }
  ]
}
```

**查看实时日志：**
```powershell
# 方法 1：直接在终端查看
# 在运行 uvicorn 的终端中直接看到输出

# 方法 2：查看日志文件
Get-Content ai-service/logs/app.log -Wait -Tail 50

# 方法 3：Docker 日志
docker-compose logs -f ai-service
```

### 代码修改约定
- 日志使用 `app.core.logging.logger`，不使用 print
- API 响应统一使用 `ChatResponse` schema
- 错误处理必须包含 `exc_info=True`
- 修改 LangGraph workflow 需要更新 `ConversationState`

### Windows 常用快捷键

**VS Code 相关：**
- `Ctrl + `` - 打开/关闭集成终端
- `Ctrl + Shift + P` - 打开命令面板
- `Ctrl + P` - 快速打开文件
- `Ctrl + Shift + F` - 全局搜索
- `F5` - 启动调试

**Windows Terminal 相关：**
- `Ctrl + Shift + T` - 新建标签页
- `Ctrl + Shift + W` - 关闭标签页
- `Ctrl + Tab` - 切换标签页
- `Alt + Enter` - 全屏模式

**PowerShell 相关：**
- `Ctrl + C` - 中断当前命令
- `Ctrl + L` - 清屏（或使用 `Clear` 命令）
- `↑ / ↓` - 浏览历史命令
- `Tab` - 自动补全

---

## Windows 使用总结

在 Windows 上使用 Claude Code 的关键要点：

✅ **推荐配置**：
1. 使用 VS Code + 集成终端（最佳体验）
2. 或使用 Windows Terminal + PowerShell
3. 安装 Git for Windows（获得 Git Bash）
4. 安装 Docker Desktop（用于容器化服务）

⚠️ **注意事项**：
1. 路径使用 `\` 分隔符，或用引号括起来
2. PowerShell 可能需要调整执行策略
3. Python 虚拟环境激活使用 `.ps1` 或 `.bat` 脚本
4. 环境变量设置使用 `$env:` 语法

💡 **最佳实践**：
1. 使用 Tab 补全（自动处理路径格式）
2. 在 VS Code 中直接使用 Claude Code 扩展
3. 优先使用 PowerShell 而非 CMD
4. 配置好 `$PROFILE` 以持久化环境变量

---

**提示**: Claude Code 会根据项目中的 `CLAUDE.md` 和 `.github/copilot-instructions.md` 自动调整行为，确保这些文件保持最新。

**快速参考**：
- VS Code 扩展: 在扩展市场搜索 "Claude Code"
- 安装命令: `npm install -g @anthropic-ai/claude-code`
- 启动命令: `claude`
- 帮助命令: `/help`

---

## WSL 专项指南

### 为什么推荐使用 WSL？

WSL（Windows Subsystem for Linux）是 Windows 上运行 Linux 环境的最佳方案，特别适合本项目：

✅ **优势**：
- 原生 Linux 环境，完全兼容所有 Unix 命令
- Python 虚拟环境管理更简单（source activate）
- Docker 集成更好（Docker Desktop 直接支持 WSL2）
- 路径和权限问题更少
- 性能通常比 Windows 原生更好

### WSL 安装与配置

**1. 安装 WSL 2（推荐 Ubuntu 22.04 或 24.04）**
```powershell
# 在 Windows PowerShell（管理员）中运行
wsl --install

# 或者指定发行版
wsl --install -d Ubuntu-22.04

# 安装完成后重启计算机
```

**2. 更新 WSL 到最新版本**
```powershell
wsl --update
```

**3. 配置 WSL 用户和权限**
```bash
# 首次启动会提示创建用户名和密码
# 之后在 WSL 终端中：

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基本工具
sudo apt install -y build-essential curl wget git vim
```

### WSL 环境配置

**安装 Node.js（用于 Claude Code）**
```bash
# 使用 NodeSource 仓库安装最新 LTS 版本
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

# 验证安装
node --version
npm --version
```

**安装 Python（项目需要）**
```bash
# Ubuntu 通常自带 Python3，如需安装
sudo apt install -y python3 python3-pip python3-venv

# 验证安装
python3 --version
pip3 --version
```

**配置 Git**
```bash
# 安装 Git
sudo apt install -y git

# 配置用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 配置换行符处理（Windows + Linux 混合环境）
git config --global core.autocrlf input
```

### 在 WSL 中使用本项目

**访问 Windows 文件系统**
```bash
# WSL 自动挂载 Windows 驱动器到 /mnt/
# D: 盘 → /mnt/d/
cd /mnt/d/zenking_work/metahuman_work/ZengKingMorphe

# 查看项目文件
ls -la
```

**创建 Python 虚拟环境**
```bash
cd /mnt/d/zenking_work/metahuman_work/ZengKingMorphe

# 虚拟环境已存在，直接激活
source .venv/bin/activate

# 或直接使用项目 Python 解释器
/mnt/d/zenking_work/metahuman_work/ZengKingMorphe/.venv/bin/python --version

# 安装依赖（如需要）
pip install -r ai-service/requirements.txt
```

**使用 Docker**
```bash
# 确保已安装 Docker Desktop for Windows
# 在 Docker Desktop 设置中启用 "Use the WSL 2 based engine"

# 启动服务
docker-compose up -d mongodb elasticsearch chroma

# 查看日志
docker-compose logs -f

# 查看服务状态
docker-compose ps
```

**启动 AI 服务**
```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 进入 ai-service 目录
cd ai-service

# 启动服务
uvicorn main:app --reload --port 8000
```

**运行测试**
```bash
# 在项目根目录
cd /mnt/d/zenking_work/metahuman_work/ZengKingMorphe

# 激活虚拟环境
source .venv/bin/activate

# 运行测试
pytest tests/

# 运行特定测试并显示输出
pytest tests/test_web_search.py -v -s
```

### VS Code + WSL 集成

**在 WSL 环境中打开 VS Code**
```bash
# 在 WSL 终端中项目目录下执行
cd /mnt/d/zenking_work/metahuman_work/ZengKingMorphe
code .

# VS Code 会自动检测 WSL 环境并安装扩展
# 终端会自动使用 WSL 环境
```

**VS Code Remote - WSL 扩展功能**：
- 文件在 WSL 中编辑，但界面在 Windows
- 智能提示和调试完全支持
- 无缝访问 Windows 和 Linux 文件系统
- 扩展自动安装在 WSL 环境中

### WSL 环境变量配置

**编辑 ~/.bashrc**
```bash
# 打开配置文件
nano ~/.bashrc

# 添加以下内容到文件末尾

# Claude Code API Key
export ANTHROPIC_API_KEY="your-api-key"

# Python 相关
export PYTHONPATH="${PYTHONPATH}:/mnt/d/zenking_work/metahuman_work/ZengKingMorphe"

# 项目别名
alias morphe="cd /mnt/d/zenking_work/metahuman_work/ZengKingMorphe"
alias activate_morphe="cd /mnt/d/zenking_work/metahuman_work/ZengKingMorphe && source .venv/bin/activate"
alias pymorphe="/mnt/d/zenking_work/metahuman_work/ZengKingMorphe/.venv/bin/python"

# 保存并退出（Ctrl+X, Y, Enter）

# 重新加载配置
source ~/.bashrc
```

### WSL 常用命令

**Windows 和 WSL 互操作**
```bash
# 从 WSL 调用 Windows 命令
explorer.exe .                # 在文件资源管理器中打开当前目录
code .                        # 用 VS Code 打开当前目录
cmd.exe /c echo "hello"       # 运行 Windows CMD 命令
powershell.exe Get-Process    # 运行 PowerShell 命令

# 从 Windows 调用 WSL 命令
wsl ls /mnt/d/                # 在 PowerShell 中运行 Linux 命令
wsl -d Ubuntu-22.04 ls        # 在特定发行版中运行
```

**WSL 管理**
```powershell
# 在 PowerShell/WSL 中
wsl --list --verbose          # 查看所有 WSL 发行版
wsl --shutdown                # 关闭所有 WSL 实例
wsl --terminate Ubuntu-22.04  # 终止特定发行版
wsl --export Ubuntu distro.tar # 导出发行版
```

### WSL 性能优化

**将项目代码放在 WSL 文件系统中**（推荐）
```bash
# 跨文件系统访问（Windows ↔ WSL）会有性能损失
# 推荐将频繁访问的项目放在 WSL 文件系统中

# 复制项目到 WSL 主目录
cp -r /mnt/d/zenking_work/metahuman_work/ZengKingMorphe ~/morphe
cd ~/morphe

# 或者创建符号链接
ln -s /mnt/d/zenking_work/metahuman_work/ZengKingMorphe ~/morphe
```

**配置 .wslconfig（可选）**
```powershell
# 在 Windows 用户目录下创建文件
notepad $env:USERPROFILE\.wslconfig

# 添加以下内容以优化性能
[wsl2]
memory=16GB
processors=8
swap=2GB
localhostForwarding=true

# 保存后重启 WSL
wsl --shutdown
```

### WSL 故障排除

**问题 1：权限问题**
```bash
# 修复文件权限
sudo chown -R $USER:$USER ~/.venv
chmod +x .venv/bin/activate
```

**问题 2：Docker 无法连接**
```bash
# 检查 Docker Desktop WSL 集成
# Docker Desktop → Settings → Resources → WSL Integration
# 确保启用了你的 WSL 发行版

# 测试连接
docker ps
```

**问题 3：网络问题**
```bash
# WSL2 使用虚拟网络，可能需要配置防火墙
# 如果无法访问 localhost 服务，尝试使用 Windows IP

# 获取 Windows IP
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
```

**问题 4：磁盘空间不足**
```bash
# 查看 WSL 磁盘使用
df -h

# 清理包缓存
sudo apt clean
sudo apt autoremove

# 扩展 WSL 虚拟磁盘（需要在 Windows PowerShell 中操作）
wsl --shutdown
# 在磁盘管理中扩展 VHDX 文件
```

### WSL 使用建议

**最佳实践**：
1. ✅ 使用 VS Code Remote - WSL 扩展进行开发
2. ✅ 将频繁构建的项目放在 WSL 文件系统（~）中
3. ✅ 在 ~/.bashrc 中配置常用别名和环境变量
4. ✅ 使用 Windows Terminal 并配置多个 WSL 配置文件
5. ✅ 定期更新 WSL 和发行版包

**避免**：
1. ❌ 不要在 WSL 中使用 Windows 路径进行频繁的文件操作（性能差）
2. ❌ 不要在 WSL1 中运行 Docker（只支持 WSL2）
3. ❌ 不要在 WSL 中直接访问 Windows 系统文件

---

**总结**：WSL 是 Windows 上运行 Linux 工具的最佳方案，本项目完全支持 WSL 环境。对于需要 Python、Docker、Node.js 的复杂项目，WSL 通常比纯 Windows 环境更稳定、性能更好。
