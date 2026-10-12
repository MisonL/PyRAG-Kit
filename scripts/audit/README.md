# 变异审计工具

`scripts/audit/` 把一次性变异扫描收口成可复现的三步工具链：

```bash
# 1. 枚举候选（路径相对仓库根，列是 UTF-8 字节偏移）
uv run python scripts/audit/list_sites.py --kinds eqneq src/ > /tmp/eqneq.txt

# 2. 扫描：无 selector 时对每个变异跑全量套件，-x 让被杀变异快速结束
#    启动时会先跑一次**未变异**基线套件；基线不绿（有预先存在的失败）会中止整轮，
#    否则红套件会把真存活点误判成 KILLED（假绿）。加 --no-baseline 跳过此门。
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
- `eqneq`：`==` ↔ `!=`；
- `cmpdir`：`<` ↔ `>`、`<=` ↔ `>=`（方向反转）；
- `boolc`：`True` ↔ `False`（布尔字面量互换）；
- `arith`：`+` ↔ `-`、`*` ↔ `/`（加减互换、乘除互换；只收数值二元运算，
  字符串字面量参与的拼接不算位点，站点键取左操作数结尾以区分链式表达式）。
  AST 无法静态区分类型，因此 `Path / x` 这类**路径拼接除法**也会进位点——其
  变异体（`Path * str`）在运行期抛 `TypeError` 而被杀，只让 KILLED 计数虚高，
  不产生假结论。

`expected` 可为 `KILLED`、`SURVIVED`、`MUTFAIL` 或 `COLLECT_ERROR`。验证器会在启动时同步每个已注册且干净的 worktree 到主仓 HEAD；先跑未变异**基线套件**，不绿即中止（`--no-baseline` 可跳过，自担风险）；pytest 输出里出现 `No module named pytest`/`INTERNALERROR` 等基础设施故障标记时归为非判定（退出码 1 的「环境不可用」与「测试失败」同码）；选择器预检按 AST 收集函数名（顶格 def、类方法、async def 都识别，旧 `^def` 正则只认顶格）；pytest 退出码 4/5 不会被当成红测；源码行号漂移只在窗口内找到唯一最近节点时自动修正，否则明确报 `MUTFAIL`。每条变异都在 `finally` 中还原，结束时再次检查 worktree 清洁。

**读源码文本的测试不参与判定**：套件里有一批测试断言的是「源码本身长什么样」（夹具坐标核对、重复装饰符扫描、`importlib.import_module` 存在性等，清单见 `verify_sites._SOURCE_INSPECTING_TESTS`）。变异期间源码处于变异态，这类测试只要被扫到就必红，测的却不是被测行为——把它们算进「有没有测试失败」会让假 KILLED。实测 green39 夹具里 4 个 SURVIVED 位点落在 `test_green39_fixture_sites_resolve_at_head` 引用的节点上，因此全部被误报 KILLED。这些测试只在**未变异**的基线上跑（普通 pytest 与 `--selftest` 均不受影响）；把它们放进排除列表前要逐个 `--deselect` 实测「只剩它红」，`--deselect` 对不存在的 node id 是静默通过的。

位点不明确或基础设施出错时不会自动「猜成 KILLED」。先运行自检：

```bash
uv run python scripts/audit/verify_sites.py --selftest
```

`regressions/green39.txt` 是已人工复核的验收夹具，不接入默认 pytest；它用于验证工具链本身的漂移、选择器、退出码和回归比对保护仍有效。**坐标已基线化到某个 HEAD**：源码继续演化会让坐标失效（报 MUTFAIL）——`tests/scripts/test_audit_tooling.py::test_green39_fixture_sites_resolve_at_head` 会把失效的位点列出来，届时按当轮源码重新定位或删除对应行。
