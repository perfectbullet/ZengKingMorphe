# Troubleshooting

## Common Issues

### Ollama responses are slow

Model being unloaded from memory. Increase `OLLAMA_KEEP_ALIVE_INTERVAL` or verify keep-alive service is running.

-> Details: `docs/Ollama模型保活方案.md`

### Embedding API errors

Incorrect `EMBEDDING_BASE_URL` or `EMBEDDING_TYPE` mismatch. Verify embedding service is running.

-> Details: `docs/Ollama-Embedding集成说明.md`

### ChromaDB connection errors

Chroma container not running. `docker-compose ps chroma` - should show port 8200.

Fix: `docker-compose up -d chroma`

### "No module named" errors

Not using virtual environment. Always activate conda: `conda activate morphe` or use full path `/home/zj/miniconda3/envs/morphe/bin/python`.

### Authentication issues

Middleware exists at `ai-service/app/api/middleware/auth.py` but `settings.api_keys` list is empty (auth currently DISABLED).

## Error Recovery (Harness Pattern)

When Agent encounters errors during development:

1. **Run the failing test first** - understand the exact error:
   ```bash
   python -m pytest tests/test_xxx.py -v
   ```

2. **Read the error message carefully** - don't guess. Look at traceback, file path, line number.

3. **Check recent changes** - what was modified that could cause this:
   ```bash
   git diff HEAD~1
   git log --oneline -5
   ```

4. **Fix incrementally** - make one small change, re-test. Don't shotgun multiple fixes.

5. **If stuck, revert to last working state**:
   ```bash
   git stash  # save current work
   # verify the last commit works
   python -m pytest tests/ -v
   ```

## Performance Issues

-> See `docs/性能优化总结.md` for optimization history
-> See `docs/性能优化方案-第二阶段.md` for planned improvements
-> See `docs/性能优化效果验证.md` for benchmark results

## Deployment Issues

-> See `docs/部署指南.md` for full deployment instructions
