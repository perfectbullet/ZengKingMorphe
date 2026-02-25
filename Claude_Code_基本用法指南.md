# Claude Code 基本用法指南

## 简介

Claude Code 是 Anthropic 官方的 CLI 工具，用于在命令行环境中与 Claude AI 交互，辅助软件开发任务。

## 推荐环境：WSL + Ubuntu

本项目推荐在 **WSL (Windows Subsystem for Linux)** + **Ubuntu** 环境中使用 Claude Code。

### 为什么选择 WSL？

- 原生 Linux 环境，完全兼容所有 Unix 命令
- Python 虚拟环境管理更简单（source activate）
- Docker 集成更好（Docker Desktop 直接支持 WSL2）
- 路径和权限问题更少
- 性能通常比 Windows 原生更好

## WSL 环境安装与配置

### 1. 安装 WSL 2

```powershell
# 在 Windows PowerShell（管理员）中运行
wsl --install

# 或者指定 Ubuntu 发行版
wsl --install -d Ubuntu-22.04

# 安装完成后重启计算机
```

### 2. 更新 WSL 到最新版本

```powershell
wsl --update
```

### 3. 首次配置 WSL

```bash
# 首次启动会提示创建用户名和密码
# 之后在 WSL 终端中更新系统
sudo apt update && sudo apt upgrade -y

# 安装基本工具
sudo apt install -y build-essential curl wget git vim
```

### 4. 安装 Node.js（Claude Code 需要）

```bash
# 使用 NodeSource 仓库安装最新 LTS 版本
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

# 验证安装
node --version
npm --version
```

### 5. 安装 Python（项目需要）

```bash
# Ubuntu 通常自带 Python3，如需安装
sudo apt install -y python3 python3-pip python3-venv

# 验证安装
python3 --version
pip3 --version
```

### 6. 配置 Git

```bash
# 安装 Git
sudo apt install -y git

# 配置用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 配置换行符处理
git config --global core.autocrlf input
```

## 安装 Claude Code

```bash
# 使用 npm 安装
npm install -g @anthropic-ai/claude-code

# 验证安装
claude --version
```

## 启动 Claude Code

### 方法 1：在 WSL 终端中启动

```bash
# 切换到项目目录
cd /home/zj/ZengKingMorphe

# 启动 Claude Code
claude
```

### 方法 2：使用 Windows Terminal（推荐）

```bash
# 打开 Windows Terminal
# 切换到 WSL（Ubuntu）标签页
# 或使用快捷键打开新的 WSL 标签

# 切换到项目目录
cd /home/zj/ZengKingMorphe

# 启动 Claude Code
claude
```

### 方法 3：VS Code 集成（最推荐）

```bash
# 在 WSL 终端中项目目录下执行
cd /home/zj/ZengKingMorphe
code .

# 安装 "Claude Code" VS Code 扩展
# 打开命令面板（Ctrl + Shift + P）
# 输入 "Claude Code: Start"
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
"运行 pytest tests/"

# 安装依赖
"安装 lodash 和 axios"

# Git 操作
"查看当前 git 状态"
```

### 4. 代码解释

```bash
# 解释代码功能
"解释 src/utils.py 中的 parse_data 函数"

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

## 配置选项

### 环境变量设置

**临时设置（当前会话）**
```bash
# 在 ~/.bashrc 中添加
export ANTHROPIC_API_KEY="your-api-key"
export CLAUDE_MODEL="claude-sonnet-4.5"
export CLAUDE_MAX_CONCURRENT_TASKS="5"
```

**永久设置**
```bash
# 编辑 ~/.bashrc
nano ~/.bashrc

# 添加以下内容
export ANTHROPIC_API_KEY="your-api-key"
export CLAUDE_MODEL="claude-sonnet-4.5"

# 重新加载配置
source ~/.bashrc
```

### 配置文件

**配置文件位置**：
```bash
# 用户配置路径
~/.claude/config.json

# 查看配置路径
echo ~/.claude/config.json
```

**配置文件内容**：
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

**创建配置文件**：
```bash
# 创建配置目录
mkdir -p ~/.claude

# 创建配置文件
nano ~/.claude/config.json
```

### 便捷别名配置

在 `~/.bashrc` 中添加项目相关别名：

```bash
# 打开配置文件
nano ~/.bashrc

# 添加以下内容到文件末尾

# Python 相关
export PYTHONPATH="${PYTHONPATH}:/home/zj/ZengKingMorphe"

# 项目别名
alias morphe="cd /home/zj/ZengKingMorphe"
alias activate_morphe="cd /home/zj/ZengKingMorphe && source .venv/bin/activate"
alias pymorphe="/home/zj/ZengKingMorphe/.venv/bin/python"

# 保存并退出（Ctrl+X, Y, Enter）

# 重新加载配置
source ~/.bashrc
```

## Hooks 自动化

配置自动化工作流：

```json
{
  "hooks": {
    "pre-commit": "pytest tests/ && flake8 app/",
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

## 本项目特殊用法

### 启动开发环境

**方法 1：使用 Docker（推荐）**
```bash
# 切换到项目目录
cd /home/zj/ZengKingMorphe

# 启动数据库服务
docker-compose up -d mongodb elasticsearch chroma

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

**方法 2：本地运行 AI 服务**
```bash
# 1. 先启动数据库
docker-compose up -d mongodb elasticsearch chroma

# 2. 进入项目目录
cd /home/zj/ZengKingMorphe

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 进入 ai-service 目录
cd ai-service

# 5. 启动服务
uvicorn main:app --reload --port 8000

# 6. 访问 API 文档
# 浏览器打开: http://localhost:8000/docs
```

**方法 3：完整 Docker 部署**
```bash
# 启动所有服务（包括 AI 服务）
docker-compose up -d

# 查看所有日志
docker-compose logs -f

# 停止所有服务
docker-compose down
```

### 运行测试

```bash
# 确保在项目根目录
cd /home/zj/ZengKingMorphe

# 激活虚拟环境
source .venv/bin/activate

# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_web_search.py

# 运行测试并显示输出
pytest tests/ -v -s

# 运行测试并生成覆盖率报告
pytest tests/ --cov=app --cov-report=html
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
```bash
# 方法 1：直接在终端查看（运行 uvicorn 的终端）

# 方法 2：查看日志文件
tail -f ai-service/logs/app.log

# 方法 3：Docker 日志
docker-compose logs -f ai-service
```

### 代码修改约定

- 日志使用 `app.core.logging.logger`，不使用 print
- API 响应统一使用 `ChatResponse` schema
- 错误处理必须包含 `exc_info=True`
- 修改 LangGraph workflow 需要更新 `ConversationState`
- Python 命令必须使用虚拟环境：`.venv/bin/python` 或先 `source .venv/bin/activate`

## VS Code + WSL 集成

### 在 WSL 环境中打开 VS Code

```bash
# 在 WSL 终端中项目目录下执行
cd /home/zj/ZengKingMorphe
code .

# VS Code 会自动检测 WSL 环境并安装扩展
# 终端会自动使用 WSL 环境
```

### VS Code Remote - WSL 扩展功能

- 文件在 WSL 中编辑，但界面在 Windows
- 智能提示和调试完全支持
- 无缝访问 Windows 和 Linux 文件系统
- 扩展自动安装在 WSL 环境中

## WSL 常用命令

### Windows 和 WSL 互操作

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

### WSL 管理

```bash
# 查看所有 WSL 发行版
wsl --list --verbose

# 关闭所有 WSL 实例
wsl --shutdown

# 终止特定发行版
wsl --terminate Ubuntu-22.04

# 导出发行版
wsl --export Ubuntu distro.tar
```

## WSL 性能优化

### 配置 .wslconfig（可选）

```powershell
# 在 Windows PowerShell 中创建文件
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

### 项目代码位置建议

```bash
# 跨文件系统访问（Windows ↔ WSL）会有性能损失
# 推荐将频繁访问的项目放在 WSL 文件系统中

# 复制项目到 WSL 主目录
cp -r /home/zj/ZengKingMorphe ~/morphe
cd ~/morphe

# 或者创建符号链接
ln -s /home/zj/ZengKingMorphe ~/morphe
```

## 故障排除

### 问题 1：权限问题

```bash
# 修复文件权限
sudo chown -R $USER:$USER ~/.venv
chmod +x .venv/bin/activate
```

### 问题 2：Docker 无法连接

```bash
# 检查 Docker Desktop WSL 集成
# Docker Desktop → Settings → Resources → WSL Integration
# 确保启用了你的 WSL 发行版

# 测试连接
docker ps
```

### 问题 3：网络问题

```bash
# WSL2 使用虚拟网络，可能需要配置防火墙
# 如果无法访问 localhost 服务，尝试使用 Windows IP

# 获取 Windows IP
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
```

### 问题 4：磁盘空间不足

```bash
# 查看 WSL 磁盘使用
df -h

# 清理包缓存
sudo apt clean
sudo apt autoremove
```

### 问题 5：Claude Code 命令不起作用

```bash
# 检查 Claude Code 是否正确安装
claude --version

# 重新安装
npm uninstall -g @anthropic-ai/claude-code
npm install -g @anthropic-ai/claude-code

# 清除 npm 缓存
npm cache clean --force
```

### 问题 6：中文文件名或路径

```bash
# 确保终端使用 UTF-8 编码
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 添加到 ~/.bashrc 持久化
echo 'export LANG=zh_CN.UTF-8' >> ~/.bashrc
echo 'export LC_ALL=zh_CN.UTF-8' >> ~/.bashrc
```

## 常用快捷键

### VS Code 相关

- `Ctrl + `` - 打开/关闭集成终端
- `Ctrl + Shift + P` - 打开命令面板
- `Ctrl + P` - 快速打开文件
- `Ctrl + Shift + F` - 全局搜索
- `F5` - 启动调试

### Windows Terminal 相关

- `Ctrl + Shift + T` - 新建标签页
- `Ctrl + Shift + W` - 关闭标签页
- `Ctrl + Tab` - 切换标签页
- `Alt + Enter` - 全屏模式

### Bash 相关

- `Ctrl + C` - 中断当前命令
- `Ctrl + L` - 清屏（或使用 `clear` 命令）
- `Ctrl + A` - 移动到命令行首
- `Ctrl + E` - 移动到命令行尾
- `Ctrl + R` - 搜索历史命令
- `Tab` - 自动补全

## 最佳实践

### 1. 提供清晰的上下文

```bash
# ✅ 好的做法
"在 app/services/user.py 中，修改 UserService 类的 login 方法，添加 JWT token 生成"

# ❌ 避免模糊指令
"修改登录功能"
```

### 2. 分阶段处理复杂任务

```bash
# 第一阶段：理解代码
"分析当前的认证流程"

# 第二阶段：制定计划
"如何添加 OAuth2 支持？"

# 第三阶段：实施
"按照刚才的计划实施 OAuth2"
```

### 3. 使用项目特定指令

在项目根目录创建 `CLAUDE.md`：
```markdown
# 项目约定

- 所有 Python 命令必须使用虚拟环境
- 日志使用 `app.core.logging.logger`
- API 响应统一使用 `ChatResponse` schema
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
"为 API 模块生成文档注释"
```

### 场景 5：代码审查

```bash
"审查 src/api/ 目录下的所有文件"
```

## 获取帮助

- 官方文档: https://github.com/anthropics/claude-code
- 问题反馈: https://github.com/anthropics/claude-code/issues
- 使用 `/help` 查看内置帮助

---

## 快速参考

- 安装命令: `npm install -g @anthropic-ai/claude-code`
- 启动命令: `claude`
- 帮助命令: `/help`
- WSL 项目路径: `/home/zj/ZengKingMorphe`
- Python 虚拟环境: `source .venv/bin/activate`
- 运行测试: `pytest tests/`

---

**提示**: Claude Code 会根据项目中的 `CLAUDE.md` 文件自动调整行为，确保这些文件保持最新。
