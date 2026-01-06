"""测试问候语功能导入"""
import os
os.environ["CRAG_DUMP_GRAPH"] = "0"  # 禁用图调试

from app.services.conversation_service import GREETING_KEYWORDS

print('=== Greeting Keywords Loaded Successfully ===')
print(f'Total categories: {len(GREETING_KEYWORDS)}')
for cat, keywords in GREETING_KEYWORDS.items():
    print(f'  - {cat}: {len(keywords)} keywords')

print('\n=== All Keywords ===')
for cat, keywords in GREETING_KEYWORDS.items():
    print(f'{cat}: {", ".join(keywords)}')
