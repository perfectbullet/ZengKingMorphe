#!/usr/bin/env python3
"""
批量转换logger调用从结构化格式到f-string格式
"""
import re
from pathlib import Path

WORKSPACE = Path(r"D:\zenking_work\metahuman_work\ZengKingMorphe")

def convert_logger_call(match):
    """转换单个logger调用"""
    full_match = match.group(0)
    log_level = match.group(1)
    
    # 提取消息和参数
    # 匹配 logger.info("msg", key1=val1, key2=val2, ...) 或多行版本
    msg_pattern = rf'logger\.{log_level}\(\s*["\']([^"\']*)["\']'
    msg_match = re.search(msg_pattern, full_match)
    
    if not msg_match:
        return full_match
    
    message = msg_match.group(1)
    
    # 提取所有关键字参数
    kwargs_pattern = r'(\w+)\s*=\s*([^,)]+)'
    kwargs = re.findall(kwargs_pattern, full_match)
    
    # 分离特殊参数和常规参数
    special_params = {'exc_info', 'stacklevel', 'extra'}
    regular_kwargs = []
    special_kwargs = []
    
    for key, value in kwargs:
        value = value.strip()
        if key in special_params:
            special_kwargs.append(f"{key}={value}")
        else:
            regular_kwargs.append((key, value))
    
    # 构建f-string
    if regular_kwargs:
        fstring_parts = [message] if message else []
        fstring_parts.extend([f"{key}={{{value}}}" for key, value in regular_kwargs])
        if message:
            fstring_msg = fstring_parts[0] + ": " + ", ".join(fstring_parts[1:])
        else:
            fstring_msg = ", ".join(fstring_parts)
    else:
        fstring_msg = message
    
    # 构建新的logger调用
    special_part = ", " + ", ".join(special_kwargs) if special_kwargs else ""
    new_call = f'logger.{log_level}(f"{fstring_msg}"{special_part})'
    
    return new_call


def process_file(file_path):
    """处理单个文件"""
    if not file_path.exists():
        print(f"⏭️  跳过不存在的文件: {file_path}")
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 匹配所有logger调用（包括多行）
        # 模式: logger.info("...", key=val, ...) 可能跨多行
        pattern = r'logger\.(info|error|warning|debug)\([^)]*\)'
        
        # 找到所有匹配并检查是否有关键字参数
        matches = list(re.finditer(pattern, content, re.DOTALL))
        modifications = 0
        
        for match in reversed(matches):  # 从后往前替换以保持索引
            matched_text = match.group(0)
            # 只转换有关键字参数的调用
            if '=' in matched_text and not matched_text.startswith('logger.info(f"'):
                new_text = convert_logger_call(match)
                if new_text != matched_text:
                    content = content[:match.start()] + new_text + content[match.end():]
                    modifications += 1
        
        if modifications > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {file_path.relative_to(WORKSPACE)}: {modifications} 处修改")
            return modifications
        else:
            print(f"⏭️  {file_path.relative_to(WORKSPACE)}: 无需修改")
            return 0
            
    except Exception as e:
        print(f"❌ 处理失败 {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """主函数"""
    print("=" * 70)
    print("🔄 开始转换日志格式...")
    print("=" * 70)
    
    # 所有需要处理的文件
    files_to_process = [
        # API endpoints
        "ai-service/app/api/endpoints/session.py",
        "ai-service/app/api/endpoints/employee.py",
        "ai-service/app/api/endpoints/webhook.py",
        "ai-service/app/api/endpoints/conversation.py",
        "ai-service/app/api/endpoints/knowledge_base.py",
        "ai-service/app/api/endpoints/chat.py",
        
        # Middleware
        "ai-service/app/api/middleware/rate_limit.py",
        "ai-service/app/api/middleware/auth.py",
        
        # Services
        "ai-service/app/services/rag_service.py",
        "ai-service/app/services/task_processor.py",
        "ai-service/app/services/document_service.py",
        "ai-service/app/services/conversation_service.py",
        
        # Core
        "ai-service/app/core/database.py",
        "ai-service/app/core/elasticsearch.py",
        
        # Tests
        "ai-service/tests/test_loguru_migration.py",
    ]
    
    total_modifications = 0
    total_files = 0
    
    for file_rel_path in files_to_process:
        file_path = WORKSPACE / file_rel_path
        count = process_file(file_path)
        if count > 0:
            total_files += 1
            total_modifications += count
    
    print("=" * 70)
    print(f"🎉 转换完成!")
    print(f"📊 修改文件数: {total_files}")
    print(f"📝 总修改次数: {total_modifications}")
    print("=" * 70)


if __name__ == "__main__":
    main()
