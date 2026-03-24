# MongoDB 数据库与用户创建指南

## 步骤一：进入 MongoDB 容器

在执行任何操作前，必须先进入 MongoDB 容器内部：

```bash
docker exec -it tts_mongodb bash
```

进入容器后，你会看到命令提示符变为类似 `root@<container_id>:/#` 的形式。

---

## 步骤二：连接 MongoDB Shell

在容器内执行以下命令，使用管理员账号连接：

```bash
mongosh --host localhost --port 27017 \
  -u admin -p tts_password_2024 --authenticationDatabase admin
```

成功连接后，你会看到类似以下的提示符：

```
Current Mongosh Log ID: xxxxxxxx
Connecting to:      mongodb://<credentials>@localhost:27017/?directConnection=true&authSource=admin&serverSelectionTimeoutMS=2000
Using MongoDB:      6.0.x
Using Mongosh:      2.x.x

>
```

---

## 步骤三：创建新数据库和用户

在 MongoDB Shell（`>` 提示符）中执行以下命令：

### 3.1 切换到新数据库

```javascript
use morphe_db
```

> **说明**：MongoDB 会在第一次写入数据时真正创建数据库，但我们可以先切换上下文。

### 3.2 创建数据库用户

```javascript
db.createUser({
  user: "morphe_user",
  pwd: "morphe_secure_2026",
  roles: [
    { role: "readWrite", db: "morphe_db" }
  ]
})
```

**参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `user` | `morphe_user` | 用户名 |
| `pwd` | `morphe_secure_2026` | 密码（建议生产环境更换） |
| `roles` | `readWrite` | 拥有读写权限 |

**成功输出示例**：

```
{ ok: 1 }
```

---

## 步骤四：验证创建结果

### 4.1 查看当前数据库的所有用户

```javascript
db.getUsers()
```

**预期输出**：

```javascript
[
  {
    _id: 'morphe_db.morphe_user',
    userId: new UUID("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"),
    user: 'morphe_user',
    db: 'morphe_db',
    roles: [ { role: 'readWrite', db: 'morphe_db' } ],
    mechanisms: [ 'SCRAM-SHA-1', 'SCRAM-SHA-256' ]
  }
]
```

### 4.2 验证数据库是否已创建

```javascript
show dbs
```

**预期输出**（包含 `morphe_db`）：

```
admin       140 KB
config      108 KB
local       144 KB
morphe_db    40 KB
```

> **注意**：如果数据库中没有任何数据（集合），可能不会显示。这是正常的，插入第一条数据后就会显示。

### 4.3 退出 MongoDB Shell

```javascript
exit
```

或按 `Ctrl + D`

---

## 步骤五：测试新用户连接

在容器内，使用新创建的用户进行连接测试：

```bash
mongosh --host localhost --port 27017 \
  -u morphe_user -p morp he_secure_2026 --authenticationDatabase morphe_db
```

连接成功后，可以执行以下命令验证权限：

```javascript
// 切换到 morphe_db
use morphe_db

// 插入测试文档
db.test_collection.insertOne({ created_at: new Date(), message: "连接测试成功" })

// 查询文档
db.test_collection.find()

// 删除测试集合
db.test_collection.drop()
```

---

## 附录：常用 MongoDB Shell 命令

| 命令 | 说明 |
|------|------|
| `show dbs` | 显示所有数据库 |
| `use <database>` | 切换到指定数据库 |
| `show collections` | 显示当前数据库的所有集合 |
| `db.getUsers()` | 查看当前数据库的所有用户 |
| `db.dropUser('<username>')` | 删除指定用户 |
| `db.grantRolesToUser('<user>', [{role: '<role>', db: '<db>'}])` | 为用户添加角色 |
| `exit` 或 `Ctrl+D` | 退出 MongoDB Shell |

---

## 附录：退出容器

完成所有操作后，退出容器：

```bash
exit
```

或按 `Ctrl + D`

---

## 附录：连接信息汇总

| 项目 | 值 |
|------|-----|
| 数据库名 | `morphe_db` |
| 用户名 | `morphe_user` |
| 密码 | `morphe_secure_2026` |
| 主机 | `localhost` (容器内) / `192.168.8.233` (外部访问) |
| 端口 | `27017` |
| 认证数据库 | `morphe_db` |

**连接 URI 示例**：

```
mongodb://morphe_user:morphe_secure_2026@localhost:27017/morphe_db?authSource=morphe_db
```

---

## 常见问题

### Q1: 提示 "Authentication failed"

**原因**：用户名、密码或认证数据库不正确。

**解决**：检查 `--authenticationDatabase` 参数，普通用户应指定为创建用户时的数据库名（`morphe_db`），而非 `admin`。

### Q2: show dbs 看不到新数据库

**原因**：MongoDB 在有数据前不会真正创建数据库。

**解决**：插入一条测试数据后即可看到：
```javascript
use morphe_db
db.test.insertOne({ test: 1 })
show dbs
```

### Q3: 如何删除数据库

```javascript
use morphe_db
db.dropDatabase()
```

### Q4: 如何删除用户

```javascript
use morphe_db
db.dropUser('morphe_user')
```

### Q5: 如何修改用户密码

```javascript
db.changeUserPassword('morphe_user', 'new_password_here')
```
