# MongoDB 数据迁移脚本使用指南

## 功能特性

- ✅ 支持迁移单个或多个集合
- ✅ 批量写入优化性能（默认 1000 条/批次）
- ✅ 增量迁移支持（跳过已存在的文档）
- ✅ 详细的迁移进度和统计信息
- ✅ 错误处理和日志记录
- ✅ 列出源数据库所有集合

## 快速开始

### 1. 激活 Python 环境

```bash
cd /home/zj/ZengKingMorphe/ai-service
conda activate morphe
```

### 2. 查看源数据库所有集合

```bash
python scripts/mongodb_migrate.py --list
```

输出示例：
```
源数据库 'funasr' 共有 3 个集合
  1. miners: 150 个文档
  2. transcripts: 5234 个文档
  3. audio_files: 1024 个文档
```

### 3. 迁移单个集合

```bash
python scripts/mongodb_migrate.py --collection miners
```

### 4. 迁移多个集合

```bash
python scripts/mongodb_migrate.py -c miners -c transcripts -c audio_files
```

### 5. 迁移所有集合

```bash
python scripts/mongodb_migrate.py --all
```

### 6. 迁移所有集合（排除某些集合）

```bash
python scripts/mongodb_migrate.py --all --exclude system.users
```

## 命令行参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--source-uri` | | 源数据库 URI（默认从环境变量读取） |
| `--target-uri` | | 目标数据库 URI（默认从环境变量读取） |
| `--collection` | `-c` | 要迁移的集合名称（可多次使用） |
| `--all` | `-a` | 迁移所有集合 |
| `--exclude` | `-e` | 要排除的集合（仅用于 --all） |
| `--list` | `-l` | 列出源数据库所有集合 |
| `--batch-size` | `-b` | 批量写入大小（默认 1000） |
| `--force` | `-f` | 迁移前清空目标集合（危险！） |
| `--no-skip-existing` | | 不跳过已存在的文档（会覆盖或报错） |

## 环境变量配置

可以在 `.env` 或 `.env-local` 中配置默认值：

```bash
# 源数据库配置
SOURCE_URI=mongodb://funasr:funasr2026@host.docker.internal:27017/funasr?authSource=admin

# 目标数据库配置
TARGET_URI=mongodb://morphe_user:morphe_secure_2026@192.168.8.234:27017/morphe_db?authSource=morphe_db

# 批量写入大小
BATCH_SIZE=1000
```

## 默认数据库配置

| 项目 | 默认值 |
|------|--------|
| 源数据库 URI | `mongodb://funasr:funasr2026@host.docker.internal:27017/funasr?authSource=admin` |
| 目标数据库 URI | `mongodb://morphe_user:morphe_secure_2026@192.168.8.234:27017/morphe_db?authSource=morphe_db` |
| 批量写入大小 | 1000 |

## 使用场景示例

### 场景 1: 初次迁移

迁移 `miners` 集合：
```bash
python scripts/mongodb_migrate.py -c miners
```

### 场景 2: 增量迁移

重新运行迁移，会跳过已存在的文档：
```bash
python scripts/mongodb_migrate.py -c miners
```

### 场景 3: 完全重新迁移

清空目标集合后重新迁移：
```bash
python scripts/mongodb_migrate.py -c miners --force
```

### 场景 4: 自定义数据库连接

```bash
python scripts/mongodb_migrate.py \
  --source-uri "mongodb://user:pass@source-host:27017/source_db?authSource=admin" \
  --target-uri "mongodb://user:pass@target-host:27017/target_db?authSource=target_db" \
  --collection my_collection
```

## 注意事项

1. **网络连接**：确保能访问源数据库和目标数据库
   - 源数据库使用 `host.docker.internal`，需要 Docker 容器能访问宿主机
   - 如果在本地运行，可能需要改为 `localhost` 或 `127.0.0.1`

2. **认证配置**：
   - 源数据库使用 `authSource=admin`
   - 目标数据库使用 `authSource=morphe_db`

3. **增量迁移**：默认启用增量迁移（跳过已存在的文档），如需完全重新迁移，使用 `--force`

4. **批量大小**：根据文档大小调整 `--batch-size`，大文档使用较小的批量值

## 故障排查

### 连接失败

```
错误: ServerSelectionTimeoutError
```

**解决方案**：
- 检查数据库是否运行：`docker-compose ps`
- 检查网络连接：`ping host.docker.internal`
- 检查用户名密码是否正确

### 认证失败

```
错误: Authentication failed
```

**解决方案**：
- 检查 `authSource` 参数是否正确
- 确认用户有对应数据库的访问权限

### 集合不存在

```
错误: Collection does not exist
```

**解决方案**：
- 使用 `--list` 查看源数据库有哪些集合
- 检查集合名称拼写是否正确

## 输出示例

```
============================================================
开始迁移集合: miners
============================================================
源集合文档总数: 150
目标集合当前文档数: 0
处理批次 1: 150 个文档 (进度: 150/150)
  插入 150 个新文档
============================================================
集合 'miners' 迁移完成:
  源文档总数: 150
  已迁移: 150
  跳过: 0
  错误: 0
  耗时: 2.34 秒
============================================================
```
