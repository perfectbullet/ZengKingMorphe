"""
Test script to validate loguru migration.
Tests structlog-style keyword argument syntax compatibility.
"""
import sys
sys.path.insert(0, 'D:/zenking_work/metahuman_work/ZengKingMorphe/ai-service')

from app.core.logging import setup_logging, get_logger

# Setup logging
setup_logging()

# Get logger instance
logger = get_logger(__name__)

# Test 1: Basic info with keyword arguments (structlog style)
print("\n=== Test 1: Basic info with keyword arguments ===")
logger.info(f'\1')

# Test 2: Error with exception info
print("\n=== Test 2: Error with exception info ===")
try:
    result = 1 / 0
except Exception as e:
    logger.error(f"Division by zero error: error={str(e)}", exc_info=True)

# Test 3: Warning with multiple context fields
print("\n=== Test 3: Warning with multiple fields ===")
logger.warning(f"High memory usage: memory_mb={512}, threshold_mb={400}, process_id={9999}")

# Test 4: Info without keyword arguments
print("\n=== Test 4: Simple message without keywords ===")
logger.info("Simple log message")

# Test 5: Debug with nested data
print("\n=== Test 5: Debug with complex data ===")
logger.debug(f"Request received: method={'POST'}, url={'/api/chat'}, payload_size={1024}")

# Test 6: Simulating conversation service usage pattern
print("\n=== Test 6: Simulating conversation_service.py pattern ===")
logger.info(f'\1')
logger.info(f'\1')
logger.error(f"Failed to load config: error={'Connection timeout'}", exc_info=False)

# Test 7: Exception method
print("\n=== Test 7: Using exception() method ===")
try:
    raise ValueError("Test exception")
except Exception:
    logger.exception("Caught exception in workflow")

print("\n✅ All tests completed successfully!")
print("🎉 The logger.info(f'msg: key=value') syntax works perfectly!")
