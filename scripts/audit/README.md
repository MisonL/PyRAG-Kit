# 变异审计工具

`scripts/audit/` 把一次性变异扫描收口成可复现的三步工具链：

```bash
# 1. 枚举候选（路径相对仓库根，列是 UTF-8 字节偏移）
uv run python scripts/audit/list_sites.py --kinds eqneq src/ > /tmp/eqneq.txt

# 2. 扫描：无 selector 时对每个变异跑全量套件，-x 让被杀变异快速结束
uv run python scripts/audit/verify_sites.py \
  --worktrees /path/to/verify-wt,/path/to/verify2-wt \
  /tmp/eqneq.txt

# 3. 定点红验证/回归比对
# target|line|col|kind|file::test_name|KILLED
uv run python scripts/audit/verify_sites.py \
  --worktrees /path/to/verify-wt \
  scripts/audit/regressions/green39.txt
```

## 站点文件与判定

站点行格式为 `target|line|col|kind[|selector[|expected]]`。支持的 kind：

- `drop`：删除 `not`；
- `notin`：`not in` ↔ `in`；
- `isnot`：`is not` ↔ `is`；
- `eqneq`：`==` ↔ `!=`。

`expected` 可为 `KILLED`、`SURVIVED`、`MUTFAIL` 或 `COLLECT_ERROR`。验证器会在启动时同步每个已注册且干净的 worktree 到主仓 HEAD；选择器先检查文件与测试名；pytest 退出码 4/5 不会被当成红测；源码行号漂移只在窗口内找到唯一最近节点时自动修正，否则明确报 `MUTFAIL`。每条变异都在 `finally` 中还原，结束时再次检查 worktree 清洁。

位点不明确或基础设施出错时不会自动「猜成 KILLED」。先运行自检：

```bash
uv run python scripts/audit/verify_sites.py --selftest
```

`regressions/green39.txt` 是本轮已人工复核的验收夹具，不接入默认 pytest；它用于验证工具链本身的漂移、选择器、退出码和回归比对保护仍有效。
