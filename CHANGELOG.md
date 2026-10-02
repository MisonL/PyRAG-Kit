# 更新日志

所有此项目的显著更改都将记录在此文件中。

## [Unreleased]

### 修复

- **Google 消息 content 的标量回退用裸 `str()`，超大整数让 `_contents` 崩掉**（`src/providers/google.py`）。`_contents` 对既不是 `str` 也不是 `list` 的 `content` 走 `str(raw_content)` 兜底，而 `normalize_messages` 只校验 `messages` 是「对象序列」、**不校验 content 的形态**（实测 `{"role": "user", "content": 10**4301}` 原样通过），因此这是标量 content 的唯一收口。位数超限的整数让 `str()` 抛 `ValueError: Exceeds the limit`——类型对、内容却与「消息内容写错了」无关，且异常发生在 `complete`／`invoke` 的 `try` 之外（`_contents` 在调用 SDK 之前求值），用户拿到的是整数转换错误。

  同一文件的 `_normalize_finish_reason` 早已认领同一判据（非字符串走 `safe_repr` 而不是裸 `str()`），这里对齐。回归测试两条（超大整数 content 必须保住、普通标量渲染形态不变），前者在未修源码时实测变红。

- **检索参数菜单的融合策略 `default=` 未归一，未归一取值让菜单崩掉**（`src/ui/config_menu.py`）。`edit_retrieval_params` 的融合策略分支把 `chat_config["hybrid_fusion_strategy"]` 的**原始**取值当 `questionary.select(default=...)`，而候选只有 `"rrf"`／`"weighted"` 两个字面量。`SessionConfig` 的普通属性没有 `__setattr__` 守卫，嵌入方或携带异常取值的 pickle 快照可以直写该字段。实测三种收场都不在契约内：`"RRF"`／`" rrf "`（大小写／空白未归一）与 `1` 让 questionary 抛 `ValueError: Invalid 'default' value passed`，位数超限的整数抛 `ValueError: Exceeds the limit`。异常落在 `edit_retrieval_params` 内、由 `launch_config_editor` 经 `asyncio.to_thread` 调用——调用点只接 `(KeyboardInterrupt, EOFError)`，会终止整个会话，而菜单是用户唯一还能改配置的入口。

  同一函数里 `current_fusion_strategy` 的**标签**早已走 `_render_value`（上一条同类修复），`default=` 这一半却漏了——判据只做了一半。修法是新增 `_normalized_fusion_default`：复用运行期的 `_coerce_fusion_strategy` 归一（口径与 `RetrievalService` 读取路径一致，`"RRF"`／`" rrf "` 都收敛成 `"rrf"`），归一后不在候选集合里（`"bogus"`／非字符串／超限整数）退回 `None`——questionary 的「未作答」哨兵。回归测试 6 条（五种异常取值 + 一条「规范值照旧选中」的反向守卫），其中 4 条在未修源码时实测变红；修后另用**真实** questionary 复核全部形态可构造。

- **模型配置菜单把原始键直接交给 questionary，非字符串键让菜单崩掉**（`src/ui/config_menu.py`）。`edit_model_params` 把 `_configuration_keys` 返回的**原始键**当候选传给 `questionary.select`。`SessionConfig` 的 `_VALIDATORS` 明确不登记 `*_configurations`（「按原样写入，由使用方负责其内部结构」），嵌入方或携带异常取值的 pickle 快照可以放进任意可哈希键；questionary 的 `Choice.build` 只对 `str` 有特例，其余类型落到 `c.get(...)`，实测 `int`／`tuple`／`None` 键一律抛裸 `AttributeError: 'int' object has no attribute 'get'`，位数超限的整数键更早在渲染标题时抛 `ValueError: Exceeds the limit`——菜单根本显示不出来，而它正是用户唯一还能改配置的入口。

  同一处还有第二条：`default=` 传的是**原始**激活键，而候选若被渲染成字符串标题，questionary 会拿它去候选集合里找，找不到就抛 `ValueError: Invalid 'default' value passed`——激活项被删掉、或它本身是位数超限的整数时都会命中。另外 `new_llm != current_llm` 拿渲染后的字符串比较，非字符串键永不等于自身，会让「选中同一个配置」被误判成切换、白白触发一次 LLM 重载。

  修法：新增 `_configuration_choices`（候选一律包成 `questionary.Choice(title=_render_value(key), value=key)`，`value` 保留原始键）与 `_choice_default`（`default` 不在候选集合里时退回 `None`——questionary 的「未作答」哨兵，合法），切换判据改与**原始**键比较。回归测试 4 条（三种非字符串键 + 一条陈旧激活项），均在未修源码时实测变红；修后另用**真实** questionary 复核五种键形态与陈旧 `default` 全部可构造。

- **非字符串键的判据只加在顶层，嵌套层级与四个兄弟校验器仍漏**（`src/utils/security.py`、`src/providers/google.py`）。上一条修的是 `validate_secret_free_options` 的**顶层**循环，注释还专门写了「只在顶层判」。但共享递归扫描器 `find_sensitive_option_paths` 对**每一层**映射都做 `str(key)` 建路径，`normalize_option_key`／`_is_credential_key` 也各自 `str(key)`；同一判据在嵌套层级和四个兄弟校验器（`validate_secret_free_payload`／`validate_secret_free_request_overrides`／`validate_secret_free_resource_kwargs`／`validate_secret_free_resource_args`）上都没有，而它们共用同一个扫描器。

  两种收场都不在契约内：`{10**4301: "x"}` 抛 `ValueError: Exceeds the limit`——类型对、内容与字段无关；`{1: "x"}` 则**静默通过**，随后在调用方的 `", ".join(sorted(...))` 处抛裸 `TypeError: sequence item 0: expected str instance, int found`（实测 `GoogleProvider("m", {"extra_body": {1: "x"}})` 构造成功，首次请求合并 `extra_body` 时才崩）。Google 自己的 `_validate_http_options` 递归与 `_coerce_debug_config` 也各有一处 `str(key)`／`sorted(...)`。

  修法：把判据收敛到唯一的递归点 `find_sensitive_option_paths`（新增 `label` 参数，错误消息带上具体路径），让 `normalize_option_key`／`_is_credential_key` 对不可 `str()` 的键走 `_display_repr` 降级，并在 `_coerce_debug_config` 与 `_validate_http_options` 各补一条同判据。回归测试 11 条（五类校验器 × 嵌套/顶层非字符串键、三个 provider 的嵌套键、Google 的 `http_options` 与 `debug_config`、`normalize_option_key` 的降级、以及一条「合法嵌套字符串键照常」的反向守卫），其中 10 条在未修源码时实测变红。

- **Google 拒答消息在非字符串取值下抛裸 `ValueError`**（`src/providers/google.py`）。两处 `str(message or ...)`（`_extract_refusal` 与 `_stream_events_from_chunk`）对位数超限的整数抛 `ValueError: Exceeds the limit`。`field()` 明确支持**字典形态**的响应（docstring：「兼容 Pydantic SDK 对象和字典响应」），而字典里的 `prompt_feedback.message` 不受 SDK `Optional[str]` 约束，因此该形态在文档化的输入契约内可达。

  这两处的职责恰恰是**报告拒答**：在正要报出「提示被安全策略拦截」的原因时自己崩掉，用户拿到的是与该语义毫无关联的整数转换错误，而本 provider 的错误分类按 `RuntimeError` 写（`ValueError` 不在调用方的 `except` 里）。修法是字符串原样、非字符串才降级成量级描述——与 `_render_value`／`_path_text` 同一取舍（直接 `safe_repr` 会给正常字符串加引号，改掉显示形态）。同文件的 `_candidate_refusal` 用 `isinstance(value, str)` 守卫，本就安全。回归测试三条（两处崩溃各一条 + 一条「字符串原样」的反向守卫），前两条在未修源码时实测变红。

- **`ExcelLogger.log_results` 的 `query` 列未收敛，超大整数会丢掉整批记录**（`src/retrieval_test/excel_logger.py`）。`query` 是该方法的行里**唯一**没走收敛的列：其余列分别经 `document_*`／`_cell_value`／`_json_safe`，而 `query` 直接进 `row`。签名声明 `query: str`，但它是**公开方法**，且这里正是「日志是旁路、失败不该丢记录」这条契约的入口。

  实测 `query=10**4301`（位数超限的整数）时 `openpyxl` 在 `save()` 期对每个数值调 `safe_string` → `isnan(value)`，抛 `OverflowError: int too large to convert to float`。后果正是 `_cell_value` 的 docstring 已经写明要防的那一种：`save` 外的 `except Exception` 把它吞成一行日志，整个工作簿**一个字都没落盘**——同一次查询里 3 条完全正常的结果一并丢失，用户看到「召回测试完成」而日志文件是坏的（`load_workbook` 连 `[Content_Types].xml` 都找不到）。修法是让 `query` 与其余列同判据，过一遍 `_cell_value`。回归测试两条（超大整数 query 必须保住整批、字符串 query 逐字落盘），前者在未修源码时实测变红。

- **provider options 的非字符串键漏成裸 `TypeError`**（`src/utils/security.py`、`src/providers/google.py`）。`ModelDetail.options` 声明为 `dict[str, Any]`，pydantic 会强制字符串键，因此**配置来源**的 options 到不了这里。但 provider 是**公开类**——嵌入方可以直接 `JinaProvider("m", {1: "x"})`（本仓测试就这么构造 provider），而共享前置校验 `validate_secret_free_options` 此前只判容器形态与敏感键名、**放行任意键类型**。非字符串键一路穿到各 provider 的 `sorted(set(options).difference(_OPTION_KEYS))`，抛裸 `TypeError: sequence item 0: expected str instance, int found`（混合类型键则是 `'<' not supported between instances of ...`）——不是 `ValueError`，而各 provider 的其余判据都按 `ValueError` 收敛。

  实测受影响的构造路径：`JinaProvider`／`SiliconflowRerankProvider`／`GoogleProvider`／`AnthropicProvider`／`VolcengineProvider`。修法分两处：共享入口 `validate_secret_free_options` 加一条顶层键类型判据（覆盖除 Google 外的全部），Google 因为不走共享入口、自己调 `_validate_options`，在其内部单列同一判据。回归测试覆盖三个 provider × 三种非法键形态，另加一条「字符串键照常」的反向守卫，均在未修源码时实测变红。

- **「未找到 key」消息里的可选值列表在混合类型键下抛裸 `TypeError`**（`src/ui/display_utils.py`、`src/services/embedding_service.py`）。两处在 key 未命中时都用 `sorted(configurations)` 生成「可选值」列表——直接比较**原始键**。键来自可直写映射（`SessionConfig.llm_configurations`／`rerank_configurations` 的普通属性没有 `__setattr__` 守卫，`RunConfig.embedding_configurations` 虽为 frozen dataclass 但嵌入方可自拼），混合类型键（`{1: .., "b": ..}`）在排序时抛裸 `TypeError: '<' not supported between instances of 'str' and 'int'`——不是 `ValueError`。`display_utils` 这条在 `start_chat_session_async` 里于任何提示符**之前**被调用，会话随即终止；`embedding_service` 这条在检索与建库的主路径上。

  讽刺的是这条消息的**用途正是**「告诉用户可选值是什么」，却在拼消息时自己崩掉。修法是先 `safe_repr` 渲染成字符串再排序（用 `safe_repr` 而非 `str`：位数超限的整数键连 `str()` 都抛）。两处各加回归测试，均先在未修源码时实测变红。同项目的工厂侧 `_get_model_detail` 不列可选值，因此不受影响。

- **快照标记与路径配置里的符号链接环漏成裸 `RuntimeError`**（`src/utils/config.py`、`src/retrieval/snapshot_repository.py`、`src/retrieval/vdb/faiss_store.py`）。`ACTIVE_SNAPSHOT` 是磁盘上的任意文本，其内容会被拼进路径后 `resolve()`；相对路径字段（`knowledge_base_path`／`pkl_path`／`snapshot_root`／`log_path`／`cache_path`）在 `resolve_path` 里也会被 `resolve()`。而 `Path.resolve()` 对**符号链接环**抛 `RuntimeError: Symlink loop from ...`——它不是 `ValueError`。三处都只在这条判据上漏了：

  | 入口 | 修复前的收场 |
  |---|---|
  | `Settings.resolve_path`（相对路径字段） | 裸 `RuntimeError`，穿透 `get_settings` 全部四个 handler |
  | `SnapshotRepository._require_snapshot_id_within_root`（`get_active_snapshot_id` 等四个方法） | 裸 `RuntimeError` |
  | `FaissStore.load_snapshot` 的标记解析 | 裸 `RuntimeError` |

  三处都与各自的契约相反：`resolve_path` 明确拒绝非字符串与 NUL，理由都是「不能让配置错误变成裸 traceback」；`SnapshotRepository` 的其余判据（路径形态、长度、逃逸）一律抛带指引的 `ValueError`，测试注释也写明「本类的契约是带指引的 `ValueError`」；`FaissStore.load_snapshot` 的其余拒绝路径（目录未激活、不是活动快照）同样抛 `ValueError`，调用方 `VectorStoreFactory._load_existing_state` 按 `ValueError` 分类。`resolve_path` 这条尤其要紧：`get_settings()` 在 **import 期**就被调用（`log_manager.get_module_logger`），早于任何入口的 `try`，实测 `data` 是符号链接环（`data -> data`）时 `import main` 直接以裸 traceback 终止。

  同项目的两个兄弟判据早已为同一情况收敛，这次让三处对齐：`utils/security.resolve_within` 明确把 `(OSError, RuntimeError)` 收成 `ValueError`（注释写明「一条『拒绝加载』的消息变成 pathlib 的内部错误」），`utils/cleanup._reject_unsafe_cache_dir` 同源。三处各加一条回归测试（另加一条「绝对路径不做 resolve，不受影响」的反向守卫），均在未修源码时实测变红。

- **`OpenAICompatibleProvider._normalize_request_timeout` 被写了两个 `@classmethod`**（`src/providers/openai_compatible.py`）。`e95158d` 引入请求级 `timeout` 校验时叠了两个同名装饰器。当前 CPython 下它**恰好能用**——`classmethod.__get__` 对「被包装对象也是 classmethod」会委托到内层，最终绑定的仍是真实函数，实测类／7 个子类（`openai`／`ollama`／`lm-studio`／`qwen`／`deepseek`／`grok`／`siliconflow`）／实例／`super()`／`inspect.signature` 五条路径的行为与单个装饰时逐字一致，因此此前没有测试拦下它。但它是**结构缺陷**而非风格问题：其一，它依赖 CPython 的委托行为、不在语言规范里；其二，`__func__` 的类型从 `function` 变成 `classmethod`，任何按「`classmethod.__func__` 是函数」写的内省（装饰器、签名收集、`functools.wraps` 链）都会错判；其三，函数体一旦开始用到 `cls`，语义就会漂移而无人察觉。现删去多余的一行，并加两条回归测试钉住：一条 AST 不变量扫描整个 `src/`（任何函数都不得有重名装饰器），一条运行时断言 `__func__` 是函数。

- **`cleanup.py` 的 `/private` 挡了符号链接解析点而非其指向目标**：macOS 上 `/etc`／`/var`／`/tmp` 都是指向 `/private/*` 的符号链接，用户把 `cache_path` 配成这三个目录之一时，`Path("/etc").resolve()` 产出 `/private/etc`、与清单里的 `Path("/etc")` **不相等**（它没 resolve），放行删除。实测 `Settings(cache_path="/etc")` 通过，退出时 `rmtree("/etc")` 被执行（需 root 权限才真删成，普通用户会得到权限错误）。同理 `/private` 自己也会放行——用户配成它时与清单比对的是 `/private` resolve 后的自己。三条都是「清单里有、但比对逻辑接不住」。现把 `/private` **提前**到遍历清单之前单独判（它是 macOS 符号链接树的总入口，挡一层即覆盖其下全部），并让清单里每个路径都先 resolve 再比对——与 `protected_paths` 那一轮的逻辑对齐（那里早就 resolve 了）。(`src/utils/cleanup.py`)

### 修复

- **修正 `/config` 分支的异常分类泄漏**：`start_chat_session_async` 的 `/config` 分支在检测到嵌入方传入的 bot 替身缺少约定接口（`apply_config_update` / `apply_config_update_async` 都不可调用）时抛出 `AttributeError`，而同一个 `try` 的 `except (ValueError, KeyError)` 接不住它，更内层的 `except (KeyboardInterrupt, EOFError)` 也接不住——实测会一路穿透到 `finally`，**终止整个会话**。这与该块开头注释的意图（「配置应用失败**不能**终止整个会话…报错、保留原配置、回到提示符」）直接矛盾：同一个恢复路径对 `SessionConfig` 校验失败等情形都生效，唯独这条会把用户丢出会话。现已改为抛出 `ValueError`（语义上也确实是一条配置错误——嵌入方的实现缺少约定接口），从而与同 `except` 里其它失败同收场。回归测试用 AST 静态对照「最近包裹的 `try` 的 `except` 分类」与 `raise` 的类型，精确验证异常分类不再泄漏。(`src/chat/core.py`)

- **公开诊断入口对非法 `provider_name`／`role` 的收场与工厂其余入口不一致**（`src/providers/factory.py`）。`ModelProviderFactory` 有三个**公开**诊断入口——`capabilities`／`capability_report`／`protocol_status`，源码注释写明供「配置界面和诊断」使用——它们的入参直接来自调用方，此前完全没有形态守卫：

  | 入参 | 修复前的收场 |
  |---|---|
  | `provider_name=["a"]` / `{"k":1}` | 裸 `TypeError: unhashable type: 'list'/'dict'` |
  | `provider_name=10**4301` | `ValueError: Exceeds the limit (4300 digits)` |
  | `role=10**4301` | 同上（各入口开头的 `str(role)` 转换） |

  两条都不是「渠道名/角色名写错了」。同文件的 `_get_model_detail` 早已用同一判据收敛（键先判字符串再进成员判断、消息经 `safe_repr`），同一个错误的收场不该取决于走的是哪条入口。

  判据要落在**两处**才覆盖全：`_provider_info_for_role`（三入口的共同路径）以及各入口开头**早于**它的转换与推断分支——`protocol_status` 在 `role is None` 时会先对 `provider_name` 做两次字典查找、`capabilities`／`capability_report` 会先 `str(role).strip().lower()`。新增共享 helper `_normalize_role_text`（与既有的 `_normalize_protocol_text` 同一判据），四个调用点全部接上；`provider_name` 的判据在 `protocol_status` 里提到函数最前。`role=None`（自动推断）的既有语义保持不变。

- **`ChatService` 的标量入参：与同文件的 `<content>` 收敛同一判据**（`src/services/chat_service.py`）。`_build_prompt` 对检索结果里的文档字段已经收敛（`page_content=None` 时整条跳过，绝不把字面量 `内容: None` 喂给模型），但**同一次调用**的两个标量参数 `user_input` 与 `intent` 完全没有守卫。实测修复前 `user_input=None`／`5`／`["a"]` 静默生成 `用户问题: None`／`用户问题: 5`／`用户问题: ['a']`，模型会把字面量 `None` 当成一个真实的用户问题去回答；`10**4301` 则抛裸 `ValueError: Exceeds the limit (4300 digits)`。同一个「非字符串值以字面量进 prompt」的形态，文档字段那一半修了、标量这一半漏了。`identify_intent` 的 f-string 更靠前（在 `try` **之外**），位数超限时抛的甚至不是本类承诺的 `RuntimeError: 意图识别失败`。两处现按同一判据拒绝并给出字段名。

- **`embed_in_batches` 的批次规模：`max(1, ...)` 只挡下界**（`src/services/embedding_service.py`）。原实现只做 `max(1, run_config.kb_embedding_batch_size)`，不挡形态。`RunConfig` 是 `frozen dataclass`、字段无校验器，生产路径的唯一构造点 `build_run_config` 直接照搬 `Settings` 的值（那里有 `1.._MAX_KB_CHUNK_TOKENS` 的双向界），因此嵌入方自己拼一个 `RunConfig` 就能送进任意形态：实测 `"32"`／`None` 抛裸 `TypeError: '>' not supported between instances of ...`，`10**4301` 抛 `ValueError: Exceeds the limit`——两条都与「批次配错了」无关，而同文件其余判据都收敛成带字段名的 `ValueError`。

  **上界用的是本字段自己的语义上界**：初版复用了 `require_scale_field`，而它的上界是 `_MAX_RETRIEVAL_CANDIDATES`（2000，检索候选数），比本字段的 `_MAX_KB_CHUNK_TOKENS`（1,000,000）小三个数量级——实测 `kb_embedding_batch_size=2001` 被 `Settings` 正常接受，却会被那个 helper 判为「超出合理范围」。那会把一个合法配置在运行时拒掉，比原缺陷更严重，因此改为按 `Settings.validate_positive_sizes` 的同一对界判定，并加了两条对偶回归用例（2001 与 1000000 必须放行）钉住这一点。

- **`Pipeline.process` 守卫了入参，却没守卫三个阶段的返回值**（`src/etl/pipeline.py`）。`process` 第一行就校验 `document` 是 `Document`（数据问题 → `ValueError`），而 `self.extractor.extract`／`self.cleaner.clean`／`self.splitter.split` 三个**协作者**的返回值此前没有任何守卫。实测 `splitter.split` 返回 `None` 时，紧随其后的日志行 `len(final_chunks)` 抛裸 `TypeError: object of type 'NoneType' has no len()`——报错现场在日志行，消息里既没有阶段名也没有「是谁违反了契约」。

  可达性：三个协作者都是 `BaseExtractor`／`BaseCleaner`／`BaseSplitter` 的实现，而 `Pipeline.__init__` 接受任意实现（抽象基类正是为替换实现而存在的）。同项目的 `RetrievalService._require_document_list` 对「协作者返回值」已有同一处置，其 docstring 写明：这是**接口契约**问题、抛 `TypeError`（入参形态错误才是数据问题、用 `ValueError`）。现新增 `_require_chunk_list` 并接在三个阶段上，两半都补：非列表（`None`／`5`／`"abc"`）与列表里的非 `Document` 项（`[None]`／`["x"]`）。不静默丢弃不合规条目——丢弃会让「实现错了」表现为「分片变少了」。

- **子分片的 50% 比例上界缺失：上一轮只修了父分片**（`src/etl/splitters/recursive_text_splitter.py`）。`_validate_parent_sizes` 有 `overlap * 2 > size` 的比例上界，而 `_validate_hierarchical_sizes` 里的**子**分片只有 `child_overlap >= child_size` 的相等判据——`child_overlap < child_size` 成立即通过，比例上界那一条漏了。子分片器同样产出分片（每个父块都要过它一次），同一个静默放大在子侧完全存在。实测父=1500/150、子 `size=300`、一篇 15000 字符的文档：

  | 子 overlap | 子块数 | 放大倍数 |
  |---|---|---|
  | 30（10%） | 66 | 1.0x |
  | 150（50%） | 99 | 1.5x |
  | 270（90%） | 451 | 6.8x |
  | 299（99.7%） | 13211 | **200.2x** |

  全部进入 embedding 与 FAISS 写入。**配置侧同样放行**：`Settings.validate_overlap_smaller_than_size` 只判 `kb_chunk_overlap` 与 `kb_chunk_size`（子分片的两条关系按该处注释所述有意留给分片器，因为模式是运行时参数），实测 `Settings(kb_child_chunk_size=300, kb_child_chunk_overlap=299)` 被静默接受，而父分片的同一比例取值被正确拒绝。现按父侧同一判据补上子侧的比例上界；恰好 50%（`overlap * 2 == size`）照旧放行，与父侧一致。

- **`HybridReranker._weight_tuple` 同一判据的两个分支用了两种渲染**（`src/services/retrieval_service.py`）。该函数对同一句「必须是和为正的有限数值」有两个触发点：前者是 `is_double_convertible` 判单个权重失败（用 `safe_repr`），后者是**派生值** `total` 判和溢出（用裸 `!r`）。裸 `!r` 的两条后果都实测复现：

  - `HybridReranker(10**308, 10**308)`——**普通 int**，两个权重各自都在 double 范围内、只有和溢出——报错 **759 字符**，而 `safe_repr` 的截断上限是 80；这正是 `safe_repr` 的 docstring 写明的「一条校验消息不该刷屏」。
  - `repr` 自己会抛的 `int` 子类（`is_double_convertible` 走 `isinstance` 放行后，`{v!r}` 在 f-string 插值处抛）让本方法的 `ValueError` 契约变成 `ValueError: Exceeds the limit (4300 digits)` 或裸 `RuntimeError`——与「权重配错了」毫无关系，且穿透调用方的 `except ValueError` 分类。

  可达性：`HybridReranker` 是公开类，docstring 自己写明「库调用方可能绕过 `Settings` 与 `SessionConfig` 直接构造」，本方法即为此存在的最后一道兜底。两个分支现统一走 `safe_repr`。

- **`src/utils/cleanup.py` 的 `/tmp` 清单项让 `bandit -ll` 不再归零**（质量门回归，由补齐系统目录清单的那一轮引入）。bandit 的 `B108 hardcoded_tmp_directory` 按 `context.string_val.startswith("/tmp")` 匹配**字符串字面量前缀**，认不出 `Path("/tmp")` 是「禁止删除的目录清单」而非「用固定路径建临时文件」——本行既不创建也不写入任何文件，语义与 CWE-377 恰好相反。它是清单里唯一以临时目录前缀开头的项。实测 `uv run bandit -r main.py src scripts -ll` 在 HEAD 上退出码为 1（CI 的 Security scan 步骤会失败），现按仓库既有的 `# nosec <规则> - 理由` 惯例标注理由，quality gate 恢复零告警、退出码 0。

- **补齐分片重叠的比例上界（`kb_chunk_overlap <= kb_chunk_size / 2`）**。原先只判 `overlap < size`，而 LangChain 的 `RecursiveCharacterTextSplitter` 只在 `overlap > size` 时报错、**接近** size 时静默接受——每个分片只前进 `size - overlap` 个字符，冗余随比例急剧放大。实测一篇 15000 字符的文档在 `size=1500` 时：overlap=150（10%）产出 14 个分片（1.1x），overlap=1490（99%）产出 1651 个（137.6x），overlap=1499 产出 13751 个（**1145.9x**）——全部进入 embedding 与 FAISS 写入，是一次「配置看着正常、代价上千倍」的静默放大。50% 是主流 RAG 实践的通行上界（它同时保证每个分片至少有一半是新内容），且实测在该点上放大仅 1.9x。判据同时加在 `Settings.validate_overlap_smaller_than_size`（拦配置来源）与 `RecursiveTextSplitter._validate_parent_sizes`／`_validate_hierarchical_sizes`（拦直接构造，它是公开类）——两个入口必须一致。

- **修复 anthropic 与 volcengine 渠道的 5 处边界缺陷**（第二轮对抗审查，同样逐条实测复现）：

  - **Anthropic 的服务端工具被静默降级成空 schema 的普通函数工具**：`_convert_tools` 的白名单只认 `server_` 前缀与三个 2025-01-24 时代的字面量，而 SDK 里真实存在二十种服务端工具类型（`web_search_20250305`、`code_execution_20250522`、`memory_20250818`、`web_fetch_20250910`、`browser_toolset_20260801` …）。实测发出的请求体从 `{'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': 5}` 变成 `{'name': 'web_search', 'description': '', 'input_schema': {...}}`——`max_uses` 等专属字段被丢弃，服务端收到自定义函数工具而非服务端检索工具，用户看到 web_search 静默失效。现从 SDK 的 `ToolUnionParam` 联合类型**推导**集合（除普通 `ToolParam` 外每个成员的 `type` 字面量），SDK 新增工具时自动跟随；再与旧字面量取并集——SDK 会**移除**旧版本的工具模型（实测 `computer_20250124` 已被 `computer_toolset_20260801` 取代），只取推导集合会让这些仍在服务端生效的取值从「原样透传」退回「降级」，比修复前更糟。
  - **`response_format` 非 Mapping 抛裸 `AttributeError`**（两处，同一语义两个来源）：`anthropic._response_format_config` 与 `volcengine._convert_responses_format` 都直接 `.get("type")`。`options.response_format` 那一侧早有守卫，但**请求参数**来源直接进这两个函数，实测 `"json_object"`／`42`／`['a']` 都得到 `AttributeError`——不是 `ValueError`，穿透调用方的错误分类。`google.py` 的 `_response_format_values` 对同一判据给的是显式 `ValueError`，现三处对齐。
  - **Anthropic 请求级 `max_tokens` 完全无校验**：`_default_max_tokens` 对 `options.max_tokens` 有完整判据（含专门的有限性注释），而请求参数那条路原样塞进 params，由 SDK 深处的 `_calculate_nonstreaming_timeout` 执行 `maximum_time * max_tokens / 128_000` 时才炸——实测 `10**400` 抛 `OverflowError: integer division result too large for a float`（`ArithmeticError` 子类，既不落 `except ValueError` 也不落 `except RuntimeError`），`"abc"`／`[1]` 抛 `TypeError`，而 `1.5`／`-5`／`True` 静默通过、原样发给服务端。现抽出 `_require_positive_max_tokens` 供两个入口共用。
  - **Anthropic 非流式路径读不到 `refusal` 键，拒答被静默丢弃**：流式 `_stream_event` 读 `refusal or text or reason`，非流式 `_extract_result` 只读 `reason or text`。实测 `content=[{"type": "refusal", "refusal": "被拒绝"}]`、`stop_reason="end_turn"` 时非流式返回 `text=''`、`refusal=None`、`finish_reason='end_turn'`，`invoke(stream=False)` 输出 `[]`——一次**安全拦截**变成「空成功」，而流式路径正确抛 `RuntimeError: Anthropic 请求被拒绝`。`refusal` 是 SDK `RefusalBlock` 的正式字段名，现三处取值优先级逐字一致。
  - **Ark 动态资源路径在未命中触发字段时完全跳过参数校验**：`_require_provider_resource` 只在参数里出现 `_ARK_RESPONSES_VALIDATED_KWARGS` 之一时才调 `_validate_native_response_kwargs`，于是 `resources.responses.create()`／`create(store=True)`／`create(text={...})` 直达 SDK，撞上 `_utils/_utils.py` 的 `assert "model" in kwargs`，抛 **`AssertionError`**——既不落 `except ValueError` 也不落 `except RuntimeError`；同样参数走 Facade 得到的是带指引的 `ValueError`。现对创建方法（`create`／`async_create`）无条件执行校验，其余（retrieve／delete／list／子资源）仍按触发字段判断，避免把 `retrieve("r1")` 误要求提供 `model`。

- **收敛 24 处边界缺陷（异常分类泄漏、读写不对称、静默语义破坏）**。全部来自一次六维对抗审查，每条都在修复前实测复现、修复后加了回归测试（`tests/test_boundary_hardening_regressions.py`，82 个用例；回退源码后其中 58 个变红）。

  **异常分类泄漏**（守卫抛出的异常不在调用方的 `except ValueError`／`RuntimeError` 之内）：

  - 流式工具调用的累加器把网关给的 `index`／`id` 直接当字典键。`openai` SDK 的流式解析走 `construct_type`、**跳过已声明字段的 Literal/int 校验**，实测 `{"tool_calls": [{"index": [1, 2]}]}` 原样产出 `ChoiceDeltaToolCall(index=[1, 2])`（声明类型是 `int`），随后 `setdefault` 抛裸 `TypeError: unhashable type`。新增 `hashable_key_part`，不可哈希等价于「没有稳定标识」，按既有优先级退回 `id`／位置键；Responses 路径的 `_merge_response_tool_call` 与同文件 `_response_tool_keys` 已有的 `hash()` 守卫对齐。
  - `choices` 非序列时 `_raise_for_chat_response_error` 的两条判据都不命中，调用方的 `choices[0]` 抛裸 `KeyError: 0`／`TypeError`。补一条容器形态判据（流式 usage chunk 不带 `choices` 的既有语义保持不变）。
  - 请求级 `timeout` 未经校验直达 httpx，SDK 把 `OverflowError`／`TypeError` 统一包成 `APIConnectionError('Connection error.')`，而 `is_retryable_error` 按类名匹配到 `connection` 判定**可重试**——同一个注定失败的请求被重试 3 次，用户看到的是「连接错误」而不是「超时配错了」。现与构造期 `options.timeout` 共用 `normalize_client_timeout`（`httpx.Timeout` 对象原样保留）。
  - Google 的 `_convert_tools` 对畸形工具定义抛裸 `AttributeError`／`TypeError`；同一形态在 `anthropic.py` 早已转成带序号的 `ValueError`。
  - `cleanup_old_logs` 的 `os.path.exists` 对**文件**也返回 True，`log_path` 指向普通文件时抛裸 `NotADirectoryError`。写入侧对此的处置是降级为仅控制台，清理侧由退出钩子调用——异常让程序在退出路径上以裸 traceback 终止。改用 `isdir`。
  - 融合策略的显示路径：`display_utils` 与 `config_menu` 对可直写的 `hybrid_fusion_strategy` 用裸 `str()`／`.upper()`，位数超限的整数抛 `ValueError: Exceeds the limit`、非字符串抛 `AttributeError`。统一走 `_render_value`（`safe_repr` 会给字符串加引号，改掉正常取值的显示形态，不能用）。
  - `SessionConfig.__setitem__` 的组合校验读兄弟字段时不做收敛，与同文件 `set_many` 不对称：`top_k` 的兄弟是 list 时抛裸 `TypeError`；兄弟权重是 `nan` 时**静默放行**（`nan == 0.0` 恒为 False），两个权重同时失效。现复用 `set_many` 的 `_require_finite_number`／`_require_int_value`。
  - `_runtime_verified_protocols` 在容器形态校验之前就把 `options` 当 Mapping 用：`"http_options" in options` 对整数抛裸 `TypeError`、命中后 `options.items()` 抛裸 `AttributeError`。非 google 渠道走 else 分支本来就得到 `ValueError`，两个分支的分类必须一致。
  - 深层嵌套的 TOML 让 `tomllib` 抛 `RecursionError`——它继承 `RuntimeError` 而**不是** `ValueError`，`get_settings` 的四个 handler 一个都接不住，`import main` 直接以数千帧裸 traceback 终止（深度 500 即触发）。新增 `except RecursionError` 分支。

  **读写／入口校验不对称**：

  - 流式工具调用的 `arguments` 未归一化：同一个输入在流式路径交付 dict/list/int，非流式路径经 `normalize_tool_arguments` 交付 JSON 文本——调用方按字符串 `json.loads` 时拿到 dict 并抛 `TypeError`。
  - `options.max_tokens` 赋值给 `values["max_output_tokens"]` 时绕过了 `_validate_generation_count`（`-1`／`0`／`True`／`10**400` 全部原样进入 `GenerateContentConfig`）；请求级 `max_tokens`／`candidate_count` 同样无校验。
  - Google 的 `output_dimensionality` 此前只过键名白名单，取值原样透传（`-1`／`0`／`"abc"`／`1.5`／`True` 全部通过，`10**4301` 更在 `json.dumps` 时才炸）；`openai_compatible._validate_embedding_dimensions` 已在同一语义上校验。
  - `_validate_hierarchical_sizes` 只校验**子**分片器的 `overlap < size`，父分片器缺同一判据，而父分片器在两种结构模式下都会构造。LangChain 对 `overlap == size` **静默接受**（实测 size=100/ov=100 通过、ov=101 抛错）。
  - `upsert_embeddings`（写入侧）不校验向量有限性，而载入侧 `_validate_embeddings_shape` 已校验——同一份内容「写得进、读不出」：FAISS 把 NaN/inf 照单收下（`ntotal` 正常增加），那个下标在查询时静默消失，而 `save_snapshot` 会把它落盘。
  - `load_snapshot` 只把 `lexical.index` 的条数对齐 `documents`，未对齐 `semantic.index` 的行数——`documents=2` 而索引 1 个向量时加载成功、无任何提示，第 2 篇文档永远搜不到；反向则抛 `IndexError`。

  **静默语义破坏／数据丢失**：

  - Vertex 的流式 tool_call delta 字典漏写 `will_continue`（读侧专门为它写了判据、`_merge_tool_call` 的字段循环也早已列入复制名单，唯独构造 delta 时漏掉），于是 `stream_function_call_arguments` 模式下**每一个**多分片调用都抛 `RuntimeError: 缺少稳定 id/index`。用 google-genai 自带 fixture 的真实形状验证。
  - `_partial_args_to_mapping` 的 `json_path` 下标无上界，写入循环按该下标把列表填充到那么大——模型返回 `$.items[20000000]` 即分配 2000 万个元素（实测 RSS 峰值 +76MB、耗时 6.5 秒）。新增 `_MAX_PARTIAL_ARG_INDEX = 100_000`。
  - `load_toml_config` 的 `normalize_mapping` 把 `options` 内每一层嵌套字典的键小写化，破坏用户载荷：`required = ["OrderId"]` 与 `properties = { OrderId = ... }` 被拆成 `["OrderId"]` 与 `["orderid"]`，JSON Schema 的 required 与 properties 不再对应，结构化输出静默失效。现区分「选项名」（项目/SDK 词汇，照旧归一，`MAX_TOKENS` 仍有效）与「选项载荷」（原样保留）。
  - `retrieval_candidate_multiplier` 的 `int(v)` 与其它整数字段**判据不一致**：`1.9` 被静默截断成 `1`，`"1_0"` 被接受为 10、`"+7"` 被接受为 7——而 `parse_decimal_int` 的 docstring 明确写着这两类「安静的错值」要明确拒绝。改走 `_coerce_number`，与 `chat_top_k` 对齐。
  - `_key_text` 的 `safe_repr` 降级描述只保留量级，不同的超限整数塌成同一个键（实测 `10**4301` 与 `10**4301 + 1` 相同），`_merge_hybrid_results` 于是把两条不同的文档并成一条、其中一条静默消失。改按二进制补码取摘要（`to_bytes` 不经过十进制转换），符号一并编进去。
  - `_should_apply_score_threshold` 用**原始**取值与字面量 `"rrf"` 比较，而 `HybridReranker` 会先 `.strip().lower()`：`"RRF"` 走 RRF 融合（分数是排名倒数之和）却不被识别为 RRF，阈值过滤照常套用——两种量纲混在一起，结果被静默清空。归一化收敛到 `_coerce_fusion_strategy` 一处。

- **重写流式工具调用完整性校验的判据方向**：`validate_complete_tool_call` 里相邻两条判据方向相反——「必须有非空 arguments」用黑名单（`type != "custom_tool_call"`），「必须校验 JSON」却用白名单（`type == "function_call"`）。任何不在白名单里的取值于是同时满足「必须有参数」与「不必校验参数」，半截 JSON 被当作完整调用交付。四个问题同源于此：
  - **`function` 绕过校验**：流式片段的 `type` 在两条协议上取值不同——Chat Completions 与各家 OpenAI 兼容片段用 `function`（`merge_tool_call_fragment` 的补默认值与四个 provider 的片段构造都写这个值），Responses 用 `function_call`。判据只认后者，实测四条真实流式路径（OpenAI Chat Completions、Google Gemini、Anthropic Messages、Ark Chat Completions）传 `'{"order_id":'` 时都把这段文本当完整调用交付。
  - **未知取值绕过校验**：白名单方向使任何其它取值（网关改写过的 `type`、大写形式、空串）同样豁免。实测 `type="tool_call"` 与 `type="FUNCTION"` 都交付半截 JSON。这一条**原本就有**，不是前一轮引入的。
  - **非字符串 `type` 抛裸 `TypeError`**：白名单写成集合成员测试后，`list`/`dict` 触发 `TypeError: unhashable type`。可达性走 SDK 流式解析路径——`_process_response_data` 在非严格校验下用 `construct_type` 构造 chunk，**跳过已声明字段的 `Literal` 校验**，`tool_calls[].type` 写成 `["function"]` 能到达这里（实测 `construct_type` 后 `call.type` 就是 `['function']`；这与 `enum_field` 注释里那条「`extra="allow"` 保留未知**顶层**字段」是同一后果、不同机制）。
  - **Anthropic 的两条流式路径完全没有交付前校验**：`stream_events`／`astream_events` 直接 `yield` 累积结果，其余三个 provider 的同类路径都调了 `validate_complete_tool_call`。因此它在判据修好之前连「缺 arguments」都不拦。
  - 现统一为**黑名单**：只做一次类型收敛（非字符串无法与 `custom_tool_call` 相等，按函数调用处理），只有 `custom_tool_call` 豁免——它的任意文本是 Responses 的 SDK 契约（`input: str`，无 JSON 约束），保持不变。
  - 顺带收敛同一处 `json.loads` 的另两类失败：`except json.JSONDecodeError` 只接住「JSON 语法不完整」，而 `json.loads` 还会因整数位数超过 `sys.get_int_max_str_digits()`（默认 4300）抛 `ValueError`、因嵌套深度超过解释器栈抛 `RecursionError`。前者**不是** `RuntimeError`，会穿透调用方按 `RuntimeError` 做的错误分类，消息只有 `Exceeds the limit (4300 digits)`；后者虽是 `RuntimeError` 子类、类型上落在分类内，消息却只有解释器文案、没有工具调用名。两者都让一段**结构完整**的 arguments 变成与工具调用无关的解析异常，现分别收敛成带 provider 与工具名的 `RuntimeError`。

- **修复 provider key 与配置映射的形态校验**（三处，同一缺陷类）：
  - `ModelProviderFactory._get_model_detail`：`if provider_key not in configurations` 对不可哈希取值抛裸 `TypeError: unhashable type: 'list'`（不是 `ValueError`，穿透 `Chatbot.reload_llm` 的 `except Exception` 与上游 `except ValueError` 分类），消息里的 f-string 对位数超限整数抛 `ValueError: Exceeds the limit`。同一判据在 `runtime/contracts.py` 的 `_require_known_field` 里已认领过，工厂漏了。
  - `RetrievalService._get_rerank_provider` 的 `provider_key`：不可哈希取值在 `configurations.get()` 处就抛裸 `TypeError`，**早于**工厂守卫，因此工厂那侧修不到这条路径。
  - 同函数的 `configurations`：非映射时 `.get()` 抛裸 `AttributeError: 'int' object has no attribute 'get'`。
  三者来源都是可直写的 `SessionConfig.active_llm_configuration`／`active_rerank_configuration`／`rerank_configurations`，且 `rerank_enabled=True` 时每次检索都会走到。
- **补齐 `RetrievalService.retrieve` 其余字段的形态守卫**：上一轮修了 `top_k` 与候选倍率，但同一函数里还有三个字段直接读取、同样可被直写（`SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫）：
  - `score_threshold`：`coerce_score(...) >= threshold`，非数值抛裸 `TypeError: '>=' not supported between instances of 'float' and 'NoneType'`；
  - `rerank_enabled`：`if not ...` 的真值语义把 `None`／`0`／`[]` 静默当成「禁用」、把 `"no"` 当成「启用」——用户看到的是 rerank 行为与配置不符，而不是一条配置错误；
  - `hybrid_fusion_strategy`：`_should_apply_score_threshold` 先按它判分支，非字符串在那里不会报错，要等 `HybridReranker` 才被拒（报错字段名指向构造参数而非配置项）。
  现统一在检索开始前收敛（失败早于任何检索工作），并把已收敛的值透传给下游，避免下游再读原属性绕过校验。
- **修复 `RetrievalService.retrieve` 对 `top_k`／候选倍率的形态假设**：`retrieve` 直接算 `max(1, session_config.top_k) * max(1, session_config.retrieval_candidate_multiplier)`，而 `SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫——嵌入方或携带超大整数的 pickle 快照可以把这两个字段直写成任何形态，此时 `max()` 抛裸 `TypeError: '>' not supported between instances of 'NoneType' and 'int'`。这是**生产入口**（`Chatbot.chat_async` 每次提问都调它），而异常落在 `retrieve` 既有的错误分类之外。现新增 `_coerce_scale_field`：非整数形态（含 `bool`、`nan`、`inf`、超长整数）给出带字段名的 `ValueError`，非正数沿用既有的 `max(1, ...)` 语义收敛成 1（**不改行为**），并补上乘积上界。
- **修复 Google FunctionCall `partial_args` 的 `json_path` 索引解析**：`_partial_arg_tokens` 用 `re.match(r"^\[(\d+)\]")` 取索引后直接 `int(...)`，而 `\d+` 对任意长度数字串都匹配——`int("9" * 4301)` 撞上 CPython 的 `sys.get_int_max_str_digits()`（默认 4300），抛 `ValueError: Exceeds the limit`。类型虽在本函数契约内，**内容**却是整数转换问题，与「json_path 不合法」无关，而同一函数另外两条拒绝分支都带着原始 `json_path` 与指引。`json_path` 来自模型的流式 function call 分片，形态不受控。现改走 `parse_decimal_int`（解析不了返回 `None`，绝不抛），与全项目同一判据。
- **修复检索入口的参数形态校验**：`FaissStore.semantic_search` 与 `keyword_search` 的两个参数此前没有形态守卫，失败类型都穿透调用方的 `except ValueError` 分类：
  - 查询向量：`np.asarray(10**4301, dtype=np.float32)` 抛 `OverflowError: int too large to convert to float`（`ArithmeticError` 子类，**不是** `ValueError`），容器里混入这类元素同样；字符串抛 numpy 的 `could not convert string to float`，集合／生成器抛 `TypeError`。现把转换失败统一收敛成带指引的 `ValueError`。
  - 查询文本：非字符串直接进 `jieba.cut`，抛 `AttributeError: 'int' object has no attribute 'decode'`——消息指向 jieba 内部，与「参数传错了」无关。
  本函数已按同一判据修过 `_require_top_k`，这两个参数漏了：同一个「参数形态非法」的处置，同一函数里三种收场。**不逐元素预检**：`np.zeros(4, dtype=np.float32)` 是 `embed_query` 的真实输出，逐元素判有限性会把合法的 numpy 标量一并拒掉（实测 `to_finite_float(np.float32(0.1))` 返回 `None`——它只认 Python 的 int／float），因此守卫包在 `np.asarray` 外层而不是逐元素。
- **修复快照载入的两个形状校验缺口**：`FaissStore.load` 与 `load_snapshot` 只校验了 `documents` 的条目形态与 `lexical.index` 的**顶层**形态／条数，没有校验语料的**条目**形态，也没有校验 `embeddings` 的维度。实测三组都能在生产路径上穿过 `SnapshotRepository.validate_snapshot_dir`（它同样只看顶层与条数）并让坏快照被 finalize 成**活动快照**：
  - `lexical.index` 写成 `[7]`／`[10**4301]`／`[1.5]`／`[None]` 时，`BM25Okapi._initialize` 对每条语料调 `len(document)`，抛裸 `TypeError: object of type 'int' has no len()`；
  - `embeddings.npy` 是 0 维／1 维时，`_rebuild_indices` 抛 `TypeError: len() of unsized object` 与 `IndexError: tuple index out of range`；
  - legacy pkl 里 `embeddings` 是普通列表时抛 `AttributeError: 'list' object has no attribute 'shape'`。
  这些异常类型与「快照坏了」无关，也没有重建指引，而同一份文件顶层写坏时给的正是有指引的 `ValueError`。更要紧的是 `embeddings` 的形状检查发生在赋值**之后**，因此失败会留下半更新的 store——实测一次被拒的 `load` 之后 `documents` 已是 2 条而 FAISS 索引仍是旧的 1 条，「失败必须让 store 保持原状」这条契约被绕过。现新增 `_validate_embeddings_shape`（判据与写入侧 `upsert_embeddings` 的 `ndim != 2` 对齐），并在两处补齐语料条目守卫；两个边界都在赋值前校验。
- **补齐配置编辑器里权重字段的四处渲染**：上一轮修了 `top_k` 与候选倍率的菜单项／提示串，但同一函数里**权重**的四处仍是裸插值——`current_weights` 菜单串、`输入新的向量权重 (0.0-1.0, 当前: …)` 提示串、`default=str(...)`，以及更新后的 `logger.info` f-string。同一个 `chat_config` 字典、同一个「属性可被直写」前提，只修了一部分字段。四处现统一经 `safe_repr` 收敛（logger 那处顺带改成惰性 `%s` 参数，避免 DEBUG 关闭时也eagerly 求值）。
- **修复配置面板与会话启动横幅的数值渲染**：`display_chat_config` 的五个数值字段（`vector_weight`／`keyword_weight`／`retrieval_candidate_multiplier`／`top_k`／`score_threshold`）走裸 `str()` 与 f-string 插值，`markup_text` 的值渲染同样是裸 `str(value)`。`SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫（见 `_require_finite_number` 的 docstring），嵌入方或携带超大整数的 pickle 快照可以直写这些字段，于是 `str(10**4301)` 撞上 CPython 的 `sys.get_int_max_str_digits()`（默认 4300）抛 `ValueError: Exceeds the limit`。后果比「显示不好看」严重：`display_chat_config` 在 `start_chat_session_async` 里于**任何提示符之前**被调用，用户刚看到「知识快照加载成功」就收到一条整数转换错误，会话随即终止；`markup_text` 那处则让「倍率已更新为 …」的提示自己抛错。现新增 `_render_value`（与同文件的 `_path_text` 同一取舍：只在 `str()` 真的失败时才降级，正常取值形态不变），两处共用。
- **修复配置编辑器菜单与提示里的取值渲染**：`edit_retrieval_params` 把 `chat_config` 的字段直接插进 questionary 的菜单项与提示串（`Top K (当前: …)`、`候选过量招募倍率 (当前: …)`、`输入新的Top K值 (1-…, 当前: …)` 及 `default=`）。`SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫（见 `_require_finite_number` 的docstring），嵌入方或携带超大整数的 pickle 快照可以直写 `cfg.top_k = 10**4301`，随后裸插值抛 `ValueError: Exceeds the limit`——异常落在 questionary 的渲染阶段，菜单根本显示不出来，用户看到的是整数转换错误。现这些取值统一经 `safe_repr` 收敛。另加反向守卫确认裸插值确实会抛（否则前一条断言证明不了什么），并锁住「合法取值范围内上界不退化为 0」这一不变量。
- **修复越界消息自身在渲染超大整数时崩溃（三处）**：`faiss_store._require_top_k`、`recursive_text_splitter._require_optional_size`、`embedding_service._as_float32_matrix` 都是「**类型**分支用 `safe_repr`、**范围**分支裸插值 `{value}`」的同一形态——同一参数、同一份消息，只有一处防住了 CPython 的十进制位数上限。实测正要报「越界」的那一步自己抛 `ValueError: Exceeds the limit (4300 digits) for integer string conversion`，用户拿到的是整数转换错误，与参数毫无关系，也没有上界信息。三处现统一经 `safe_repr` 收敛成 `<整数，约 N 位十进制>`（与 `runtime/contracts.py` 早已在用的判据一致）。可达性各不相同：`top_k` 走 `VectorStoreBase` 的公开方法、`expected_rows` 走公开静态方法，`value` 只有**负的**超限整数才会走到范围分支（`-(10**4301) < 1` 成立，而 `10**4301 >= 1` 直接返回）——因此三个方向的用例都要写，只测正数会让 splitter 那条守卫永远绿着。
- **修复两个载入入口的半更新状态**：`load` 先把 documents／embeddings／parent_documents 赋进 `self` 再调归一化，`load_snapshot` 每读一个文件就写一次 `self`——中途任何一步失败都留下**半更新**的 store。实测一次被拒的 `load` 之后 documents 与 embeddings 是新值，而 `faiss_index` 与 `_tokenized_docs_cache` 还是旧的（索引 2 条对应 1 条文档），随后 `semantic_search` 抛 `TypeError: 'str' object does not support item assignment`，报错现场与真正的原因（载入被拒）毫无关联；调用方本可以继续用旧数据，半更新让这个选项也消失了。现改为全部解析、校验到局部变量，最后一次赋值。归一化函数相应从就地改 `self` 的 `_normalize_loaded_documents` 改为返回新值的 `_normalize_documents`，并对条目做浅拷贝——原先 `metadata.pop("parent_content")` 会写进调用方持有的那份数据。
- **修复载入路径对 `documents`／`parents` 顶层形态的裸异常**：`SnapshotRepository.validate_snapshot_dir` 已覆盖这些形态，但 `load`／`load_snapshot` 是 `VectorStoreBase` 的契约方法，直接调它们的嵌入方不经过那道校验（与 `_validate_document_shape` 同一取舍）。实测原先的收场方式都与「快照不自洽」无关：分块列表是 `int`／`None` 时抛裸 `TypeError: 'int' object is not iterable`；是 `dict`／`str` 时逐键／逐字符当成文档、报出误导性的「第 0 条文档不是映射」（`len("ab") == 2` 更是静默通过）；父块索引是 `int`／`None`／`list` 时抛 `AttributeError: 'int' object has no attribute 'items'`；legacy pkl 顶层不是映射时抛裸 `AttributeError`。现统一为带重建指引的 `ValueError`。
- **修复 `upsert_embeddings` 在校验文档形态之前就改动 store**：该方法原先只校验 `embeddings`（维度、条数），条目形态则放任 `self.documents.extend(...)` 之后由 `_rebuild_indices` 在派生索引时撞出来。实测一条非映射条目（第三方 store 的返回值、手工构造的 pkl）让 `_build_index_text` 抛 `AttributeError: 'str' object has no attribute 'get'`——异常看似「拒绝写入」，实际 documents 已多出这条孤儿：`save_snapshot` 会把它写进快照（实测盘上 `chunks.pkl` 2 条、`embeddings.npy` 只有 1 行），同一进程内 `semantic_search` 的 `indices` 与 `documents` 下标错配（top_k=2 只返回 1 条真命中），而 `keyword_search` 的 `argsort` 下标可能越过 tokenized 缓存。现把载入边界的同一份 `Document` 契约校验（原 `_validate_loaded_document`，现更名 `_validate_document_shape`）前移到写入之前，两个调用点的消息分开：载入侧仍是「请重建快照」，写入侧改为「已拒绝写入」，避免让写入侧的用户去重建一个还没写坏的旧快照。
- **修复 `_rebuild_indices` 会因 `documents` 里的脏数据永久卡死 store**：索引重建是「由 documents 派生的缓存」，却在派生期对条目形态抛异常。实测混进一条孤儿后，**每一次** `upsert_embeddings` 都在重建索引时重复抛出——连续 3 次喂完全合法的文档全部报同一个 `AttributeError`，documents 停在 5 条不再增长，向量库事实上报废。同时 `_tokenized_docs_cache` 的赋值改为整体求值后再落属性，避免异常时留下「缓存比 documents 短」的下标错配。新增 `_index_text_of`：派生期把非法形态降级为空串（该条目缺席检索，其余条目照常可用），校验交给写入侧入口。
- **修复 `load_snapshot` 直接调用时对 `lexical.index` 的条数不做校验**：该方法在 `lexical.index` 存在时**直接使用盘上的语料**、不重建 BM25，而 `keyword_search` 拿 `np.argsort(doc_scores)` 的下标去索引 `self.documents`。实测语料 3 条、文档 2 条时 `IndexError: list index out of range`；语料 1 条、文档 2 条时静默丢结果（用户看到「搜不到」而不是「快照坏了」）；顶层写成 `None` 时 `_build_bm25_index` 返回 `None` 而上述静默丢结果同样发生。生产路径已由 `SnapshotRepository.validate_snapshot_dir` 覆盖这些形态，但 `load_snapshot` 是 `VectorStoreBase` 的契约方法，直接调它的嵌入方不经过那道校验。现补上「顶层是序列且条数与 documents 对齐」这条最小不变量，并放在 `_build_bm25_index` 之前——否则 `int` 会在 `any()` 上先抛裸 `TypeError: object of type 'int' has no len()`，`str` 则会因 `len("ab") == 2` 静默通过。
- **修复检索层的派生值溢出（权重之和、RRF 分母）**：`_weight_tuple` 对两个权重各自判了 `is_double_convertible`，但它们的**和**是派生值、可以超出 double 范围——`HybridReranker(10**308, 10**308)` 构造成功，随后任意一次 `rerank()` 在 `math.isfinite(total)` 上抛 `OverflowError`（`ArithmeticError` 子类，**不是** `ValueError`）。同类的还有 `rrf_k + rank`：`rank` 来自 store 返回的文档，实测 `semantic_rank = 10**400` 让分母抛同一个异常，`retrieval_service.retrieve()` 的 HYBRID 路径整批失败。两处现分别改为 `float()` 相加（溢出表现为 `inf`，被 `isfinite` 正常判非法）与新增 `_coerce_rank`（非法 rank 视为「未参与该路检索」，与缺失时同义）。
- **修复 rerank 缓存键的兜底自身抛错且丢失区分度**：`json.dumps(default=str)` 对超大整数无效（`default=` 只对**未知类型**生效，`int` 是已知类型），而 `repr()` 兜底抛的是同一个 `ValueError`。可达性已确认：`ModelDetail.options` 是 `dict[str, Any]`，`validate_secret_free_options` 只查敏感键名、不做递归数值校验，实测 `ModelDetail(..., options={"x": 10**4301})` 被接受，随后 `rerank_enabled=True` 时每次检索都在构造缓存键处崩。现新增 `_stable_fingerprint`——不能直接用 `safe_repr`，它对渲染失败的**容器**只退回 `<dict>`，会让两份不同的 options 撞到同一个 provider 实例（比崩溃更糟）；逐元素递归到叶子并对超大整数补 `bit_length` 与 `hash`，既保住区分度又不抛。
- **修复文档去重键与父块索引的 `str()` 残留**：`_key_text` 的替换在 `retrieval_service.py` 里漏了两处——`_document_key` 的 `page_content` 兜底支与 `_deduplicate_parent_documents` 的 `parent_key`；`faiss_store.py` 的 `parent_id` 四处（`_normalize_loaded_documents`／`register_parent_documents`／`resolve_parent_content`）也全是裸 `str()`。这些都来自磁盘上的快照／legacy pkl（pickle 是二进制格式，不受 CPython 十进制位数限制），实测经真实路径复现：`chunks.pkl` 里 `metadata.parent_id = 10**4301` 时快照被激活、加载宣布成功，但每一次检索都崩在父块解析上；`_deduplicate_parent_documents` 是三条检索路径的公共出口，因此三条全中。现统一经 `safe_repr` 收敛。
- **修复快照加载的两步顺序与 `load()` 相反**：`load_snapshot` 原先「先建索引、后归一化文档」，而 `load()` 是反的。于是「`lexical.index` 是否存在」决定了用户看到哪条错误——实测同一份畸形 chunks（`page_content` 是 `int`）：有 `lexical.index` 时得到 `ValueError: 知识快照不自洽：…请重建快照。`，缺它时得到 `AttributeError: 'int' object has no attribute 'decode'`。而 `validate_snapshot_dir` 的必需文件清单里没有 `lexical.index`，缺它是合法状态。现两处顺序对齐。
- **修复 BM25 索引对「所有 token 都为空」的除零**：`_build_bm25_index` 只判外层 `not tokenized_docs`，而 `rank_bm25` 的 `_calc_idf` 在**所有 token 都为空**时也除零（`idf_sum / len(self.idf)`）。实测 `[[]]` 与 `[[], []]` 都抛 `ZeroDivisionError`——`jieba.cut("")` 返回 `[]`，因此一条 `page_content=""` 的文档就足以构造出这个形态，而空 `page_content` 在快照里合法。两种形态语义相同，现都返回 `None`。
- **修复快照校验的三处形态缺口**：`parents.pkl` 的错误消息用 `str(key)` 拼装，键来自磁盘（含 pickle 来源），实测 `{10**4301: [1, 2]}` 让**正要报「父条目形态不对」**的那一步自己抛位数错误；`lexical.index` 的 `len()` 没有形态守卫（`is not None` 只挡 `None`，顶层是 `int` 时抛裸 `TypeError`；字符串形态更隐蔽——`len("ab") == 2` 被当成 2 条语料静默通过），而同函数的 `chunks.pkl` 有完整守卫；快照 id 无长度上界，实测 300 字符的名字通过全部判据后在 `.exists()` 抛 `OSError: [Errno 63] File name too long`——而 `VectorStoreFactory._load_existing_state` 的第一步就是它，一个被外部工具写坏的标记文件会让**启动**以 `OSError` 失败。三处现分别改用 `safe_repr`、补上与 `chunks.pkl` 同源的形态守卫、加上长度上界。
- **修复 `save_snapshot` 对不可哈希 `source` 取哈希**：统计文档数时用集合推导 `{(doc.get("metadata") or {}).get("source") for doc in self.documents}`，而 `or {}` 只挡 `None`／空值，非空的非映射会留下——实测 `source=["a"]` 抛 `TypeError: unhashable type: 'list'`。这条路径正是把文档写进快照的那一步（legacy 导入分支），失败后 `finally` 清理临时目录，legacy 快照**永远无法导入**。现改用同文件已在用的 `document_source`（对非字符串安全返回 `"N/A"`）。
- **修复 `retrieval_candidate_multiplier` 的 `inf` 分支抛 `OverflowError`**：`int(v)` 只接了 `(ValueError, TypeError)`，而 `int(inf)` 抛的是 `OverflowError`——`ArithmeticError` 子类，**不是** `ValueError`。同文件其它五个整数 validator 都有 `isinstance(number, float) and not number.is_integer()` 前置守卫，只有这一处没有，而 `float('inf').is_integer()` 恰好是 `False`、走不进那个分支。触发输入是合法 TOML（`tomllib` 把 `inf`／`1e400` 都解析成 `inf`），而 `get_settings` 在**导入期**就被 `cleanup.py` 调用，早于 `run_cli` 的任何 `try`——三个 handler 全部失效，程序以裸 traceback 退出。
- **修复路径字段接受 NUL 字节**：`os.makedirs`／`open` 对含 NUL 的路径抛 `ValueError: embedded null byte`——**不是** `OSError`，因此 `log_manager` 那些「目录不可写就降级为仅控制台」的 `except OSError` 全部接不住。实测 `config.toml` 写 `log_path = "/tmp/a\u0000b"`（TOML 的 `\u0000` 转义）时，`cleanup.py` 在**模块导入期**调用 `get_module_logger`，程序连 `run_cli` 都到不了就崩。现 `resolve_path` 显式拒绝 NUL（六个路径字段共用该 validator），`log_manager` 的两处降级也一并接住 `ValueError` 作为纵深防御。
- **修复 `_reject_unconvertible_numbers` 不递归 `ndarray`**：该函数只递归 `list`/`tuple`，而 `ndarray` 与「`ndarray` 的 list」都是 `_probe_dimension` 显式支持的输入形态。`dtype=object` 的数组带着 `10**400` 直接进 `np.array(..., dtype=np.float32)`，由 numpy 抛裸 `OverflowError`——同一批数据，`list` 形态得到设计好的干净 `ValueError`，`ndarray` 形态得到 `ArithmeticError` 子类，穿透调用方的全部 `except ValueError` 分类与日志脱敏。现补上 `ndarray` 递归（只遍历 `object` dtype，数值 dtype 的取值一定在 double 范围内）。
- **修复 `cache_path` 是符号链接环时退出钩子整体失效**：`_reject_unsafe_cache_dir` 首行就 `Path(cache_dir).resolve()`，而 `resolve()` 对链接环抛 `RuntimeError: Symlink loop from ...`。该行此前没有守卫（遍历 `protected_paths` 的那轮有），于是本函数整体抛异常——而它在 atexit 里被调用，用户看到 `Exception ignored in atexit callback` 加 traceback，**清理完全没做**。链接环不是可安全删除的目标，现返回原因而不是抛。
- **修复 `get_relative_path` 的兜底分支自己抛错**：`str(absolute_path)` 抛 `ValueError: Exceeds the limit` 后，`except ValueError` 里的 `logger.warning` **用同一个值再做一次 f-string 插值**，把刚接住的异常原样重抛。兜底分支存在的唯一理由就是不再抛——它抛了，且抛在展示检索结果的最后一环，把一次成功的检索变成「LLM 异步生成出错」。实际来源是 `document_source(document)`，而文档元数据来自磁盘上的快照／legacy pkl。
- **修复激活的 provider key 缺失时抛裸 `KeyError`**：`display_chat_config` 直接写 `chat_config["llm_configurations"][active_llm]`（两处，另一处在 API Key 段落），而 `default_llm_provider`／`default_rerank_provider`／`active_*_configuration` **没有任何 validator**（`Settings` 里只是普通 `str`，`SessionConfig` 的 `_VALIDATORS` 也没登记）。实测 `default_llm_provider = "not-a-provider"`（打错字）时，调用点 `chat/core.py` 的 `start_chat_session_async` 在任何提示符之前抛裸 `KeyError`——用户刚看到「知识快照加载成功」，紧接着会话终止。同一个 key 走 `ModelProviderFactory._get_model_detail` 得到的是 `ValueError("在…配置中未找到key: …")`，现两处对齐到该契约并列出可选值。

- **修复配置校验失败的消息泄漏凭证**：`_redacted_settings_error` 只把 pydantic 的 `input` 交给脱敏，而 `msg` 原样拼接——但 validator 普遍把**原始取值**写进消息（`f"无效的日志级别: {value}"`）。实测 `CHAT_TEMPERATURE=sk-live-...` 时异常消息里原样带着该凭证，而这条消息经 `get_settings` 的 handler 抛出后进终端与日志，正是该函数 docstring 声称要挡住的那一类。现 `msg` 与 `input` 一并脱敏。
- **修复报告配置校验失败这一步自己会崩**：同一函数里写的是 `str(error.get("input"))`，而 `input` 是 pydantic 记下的原始输入值、可以是任意大的整数。实测 `Settings(chat_top_k=10**4301)`（库调用方与测试都走这条构造路径）让**报告校验失败**这一步抛 `ValueError: Exceeds the limit`——用户看到的是位数上限错误，而真正的问题是 `chat_top_k` 越界。现改用 `safe_repr`。
- **统一 26 处异常日志出口到 `safe_exception_text`**：这些站点写的是 `redact_sensitive_text(str(e))`，绕过了同为展示边界的 `safe_exception_text`——后者是前者的严格超集（脱敏 + 长度上限 + 兜住 `str(exc)` 自身的位数超限异常）。实测异常 args 含位数超限整数时，这 26 处会在**正要记录日志**时抛 `ValueError`，而 `safe_exception_text` 能正常渲染。涉及四个 provider 适配器、`factory.py`、`cleanup.py` 与 `chat_service.py`。

- **修复融合策略、向量存储类型、快照清单的非字符串取值漏出裸异常**：三处都是「先调字符串方法再判取值」的形态，非字符串抛 `AttributeError`，而契约是 `ValueError`。`HybridReranker(fusion_strategy=["rrf"])`、`VectorStoreFactory.get_vector_store(123)` 均实测复现。项目已按同一判据修过 `SessionConfig._validate_fusion_strategy` 与 `RecursiveTextSplitter.structure_mode`，这三处是同类残留。`Settings` 能拦住配置来源的非字符串，但这些是公开入口（UI 与库调用方直接调用）。
- **修复快照清单字段与文档去重键的位数超限崩溃**：`KnowledgeSnapshotManifest.from_mapping` 写的是 `str(data[key])`，`_document_key` 对元数据同样做 `str()`。清单文件与元数据都来自磁盘，而本项目**加载 pickle 格式的快照**——pickle 是二进制格式、不受 CPython 十进制位数限制（实测 `pickle.loads(pickle.dumps(10**4301))` 正常往返），因此畸形快照能让 `str()` 在正要解析清单、或正要计算去重键合并检索结果时抛 `ValueError: Exceeds the limit (4300 digits)`，异常类型与内容都与真实原因无关。现两处都经 `safe_repr` 收敛；去重键对同一取值仍稳定，不会因降级而误合并。
- **修复 rerank provider 返回值在形态校验之前就被解包**：`a, b = await provider.arerank(...)` 直接解包，`None`／单值抛裸 `TypeError`、三元组抛裸 `ValueError`——两条都在 `_validate_rerank_output`（正是「校验 provider 返回值」的边界）**之前**，用户拿到的是「无法解包」而不是「provider 返回值不合法」。现提取 `_unpack_rerank_result`，先把形态收敛成带指引的 `RuntimeError` 再进入既有校验。
- **修复召回测试的 Excel 日志初始化失败被静默吞掉**：`ExcelLogger()` 构造失败只写一行日志（且日志级别由 `log_level` 决定，配成 `ERROR` 以上连那行都看不到），之后每轮查询走 `if excel_logger:` 静默跳过——用户照常看到召回结果，以为测试记录下来了，实际一个文件都没有。而召回测试的**唯一产出**就是这份 Excel 日志。现同时用 `console.print` 显式告知，与本文件其它失败路径（数据库加载、服务初始化）的呈现方式一致。

- **修复客户端 `timeout` 缺上界，Google 的毫秒派生值抛 `OverflowError`**：四个渠道原先只判「有限正数」，没有上界。失效点不在超时值本身，而在**派生值**上——Google 是唯一把秒换算成毫秒再交给 SDK 的渠道（`_build_http_options` 的 `int(timeout * 1000)`），任何超过 `DBL_MAX/1000 ≈ 1.8e305` 的秒数都让该派生抛 `OverflowError: cannot convert float infinity to integer`。该异常是 `ArithmeticError` 的子类而**不是** `ValueError`，穿透项目全部以 `ValueError` 为契约的 handler（`get_settings`、chat 主循环、questionary validator），以裸 traceback 终止会话。输入是普通 TOML 浮点，无需任何位数技巧：`[llm_configurations.x.options] timeout = 1e306` 让配置加载成功、provider 构造成功，首次提问才炸。现新增共用的 `normalize_client_timeout`（正数、有限、上界一天，保留 `httpx.Timeout` 形态），Google／Anthropic／OpenAI 兼容／Ark 四个渠道共用；`options.http_options.timeout`（毫秒）补上同源上界——此前 `1e308` 能通过，到真实 `genai.Client(...)` 构造时才抛 pydantic 的 `Unable to parse input string as an integer`，错误消息与 Google 配置毫无关联。Ark 渠道原本接受数字字符串，该行为保留。
- **修复 Ark `count_tokens` 与其余七个 token 计数入口契约不一致**：八个 provider 的 token 计数入口都已改用 `require_token_count`，只有 Ark 的同步/异步 `count_tokens` 漏了，写的是 `int(field(item, "total_tokens", 0))`。于是 `"abc"` 抛裸 `ValueError`、`None` 抛裸 `TypeError`、位数超限的数字串抛与 token 计数无关的位数错误，且消息里连字段名都没有；`0` 默认值更把「响应不完整」静默算成「0 个 token」。现与其余七个入口共用同一 helper。
- **修复不可哈希取值让 provider 的枚举守卫抛 `TypeError`（约 30 处）**：`field(x, "type") in {"a", "b"}` 对 `list`/`dict` 抛 `TypeError: unhashable type`，而调用方按 `ValueError`/`RuntimeError` 分类错误。这些值来自网关 JSON——openai SDK 对 SSE chunk 的未知顶层字段用 `extra="allow"` 原样保留，实测用真实 SDK 解析一条 `data: {"type": ["error"]}` 的 SSE 流即可构造，`_stream_delta` 随即抛 `TypeError`，本该报「网关返回错误」的地方变成裸 `TypeError`，日志脱敏与错误分类都接不住。现新增 `enum_field` helper（非字符串收敛成空串，落在与「未知类型」相同的 else 分支上），覆盖 Responses 的 `input_image`/`input_file` 的 `detail`、流式事件的 `type`，以及 Google／Anthropic／Ark／OpenAI 兼容四个渠道的响应解析与请求转换路径。`_normalize_responses_native_item` 原有的显式 `isinstance` 守卫是这批里唯一做对的写法，本次统一到它。
- **修复错误消息插值超大整数时改抛位数错误**：`f"{10**4300}"` 撞上 CPython 的 `sys.get_int_max_str_digits()`（默认 4300）抛 `ValueError: Exceeds the limit`。`status`／`code`／`error_code`／`message`／`reason` 这些字段来自网关 JSON，而 openai SDK 能解析出 4300 位的整数（实测经真实 SDK 解析成功，`bit_length()` 为 14281），于是「网关返回了业务错误」变成「整数转字符串超限」，`RuntimeError` 变 `ValueError`、错误分类失准。现这些插值点统一走 `safe_repr`。`status` 的读取刻意**不**用 `enum_field`：它随后要进 `_chat_status_is_error` 做 `>= 400` 的数值比较，收敛成空串会让数字型业务错误（`{"status": 503}`）漏判——形态防护只加在插值处。
- **修复 Embedding `options.dimensions` 完全不校验**：该值从 `config.toml` 直达请求体，此前 `-1`、`0`、`"abc"`、`1.5`、`True` 全部原样发给远端，由服务端返回语义不明的 400；`10**4301` 更在 `json.dumps` 编码请求体时才抛位数错误。项目里 `_MAX_EMBEDDING_DIMENSION` 早就是这个量的上界（`local_hash` 与 `embedding_service` 都在用），provider 边界漏了同一判据。
- **修复 Google 整数型生成参数无校验**：`max_output_tokens`／`candidate_count` 只做键名白名单，取值原样透传给 `types.GenerateContentConfig`。实测 `options.max_output_tokens = 10**400` 时构造与调用都成功，请求体里就带着那个 401 位整数发给服务端（`httpx.MockTransport` 抓包确认）；`candidate_count = -1`／`0` 同样静默通过。现两者都判正整数并给上界。
- **修复 rerank 的 `top_n` 缺上界**：判据是 `top_n < 1`，于是 `10**400` 与 `10**4301` 都被静默接受，后者在 `json.dumps` 编码请求体时才抛位数错误——而同一份文件里 `options.timeout` 与响应 `relevance_score` 都已判过双精度范围，只有这里漏了。上界取 `_MAX_RERANK_TOP_N` 而**不是**文档数：上游 API 自行截断，`top_n` 超过文档数是合法输入。Jina 与 SiliconFlow 两个文件是逐字副本，两侧同步修改。
- **修复 Ark Classification 的动态资源路径绕过 Facade 门禁**：AGENTS.md 要求该资源「需通过显式 Facade/入口调用，不能把缺少资源挂载静默当成成功」。Facade 侧做对了（`_get_classification_resource` 在 SDK 未挂载时抛带指引的 `NotImplementedError`），但动态路径 `resources.classification.create(...)` 既不过 Facade、也不做参数白名单——实测 `恶意字段=1` 一路放行，最终拿到 SDK 内部的 `AttributeError: 'Ark' object has no attribute 'classification'`。现动态路径与 Facade 受同一约束（同一份 kwargs 白名单 + 显式资源检查），与 `_require_responses_resource` 的既有设计对齐。

- **修复非字符串取值让配置层漏出裸异常**：`log_level` 与五个路径字段的 validator 直接对传入值调用字符串方法，而 `config.toml` 的 TOML 值可以是任意类型。`log_level = 123` 抛 `AttributeError: 'int' object has no attribute 'upper'`（实测 14 帧裸 traceback），`knowledge_base_path = 123` 抛 `TypeError: expected str, bytes or os.PathLike object, not int`（实测 23 帧）。两者都**不是** `ValueError` 的子类，穿透 `get_settings` 的全部三个 handler。`log_level` 尤其要紧：它在**模块导入期**就被 `log_manager.get_module_logger` 读取，早于任何入口的 `try`，连脱敏边界都还没建立。现两处都在取值前显式判类型，统一收敛成带字段名的脱敏 `ValueError`。顺带用 59 个字段 × 9 种非法形态（531 组）扫过整个 `Settings`，确认再无漏出非 `ValueError` 的站点。
- **同一缺陷的 Anthropic 侧**：`anthropic.Anthropic(...)` 同样接收原样透传的 `timeout`／`max_retries`，且该 SDK 的重试循环也是 `for retries_taken in range(max_retries + 1)`。实测 `max_retries=-1` 时 **HTTP 请求数为 0**，最终抛 `RuntimeError: could not resolve response (should never happen)`；`max_retries=10**400` 同样被原样接受。现与另两个渠道共用 `normalize_client_retries`，`timeout` 也在构造期校验（保留 `httpx.Timeout` 形态）。Google 渠道经核实无此问题：`max_retries` 不在其 `_OPTION_KEYS` 内会被显式拒绝，`timeout` 走已校验的 `http_options` 路径。
- 修复新增回归测试自身的一处**假绿**：`monkeypatch.setattr("src.providers.anthropic.get_settings", ...)` 的字符串形式解析到的是**当前** `sys.modules` 里的模块对象，而 `tests/providers/test_factory.py` 会把 `src.providers.anthropic` 从 `sys.modules` 删除再重新导入，于是 patch 打到新模块、而 `AnthropicProvider.__init__` 仍从旧模块字典取 `get_settings`——patch 静默落空，**单独跑该文件全绿、只有全量运行才失败**。现改为 `monkeypatch.setitem(AnthropicProvider.__init__.__globals__, ...)`，打类实际读取的那个字典。
- **修复客户端级 `max_retries` 完全不校验（OpenAI 兼容渠道）**：`timeout` 与 `max_retries` 是 openai SDK 的**客户端构造参数**，项目把它们从 `options` 原样透传给 `openai.OpenAI(...)`，而 SDK 自己不做任何检查。于是 `config.toml` 的 `options.max_retries` 可以造成三类失败：**负数**时 SDK 的重试循环 `range(max_retries + 1)` 一次都不执行，HTTP 请求根本没发出去（实测 `max_retries=-1` 请求数为 **0**），最终抛的是 SDK 内部的 `AssertionError: could not resolve response (should never happen)`；**超大值**（TOML 整数无上界，实测 `10**400` 被原样存进 `client.max_retries`）等价于无上限重试；**非整数**（`"abc"`／`1.5`／`True`）则要等到真正重试时才在 `max_retries - retries_taken` 上抛 `TypeError`，报错点离配置处很远。Ark 渠道早已校验 `max_retries`，但同样**没有上界**。现在新增共用的 `normalize_client_retries`（`0` 到 `100` 的整数，接受十进制数字字符串），两个渠道共用同一套判据，并在**构造期**校验 `timeout`（拒绝非有限、非正数与非法类型，保留 SDK 支持的 `httpx.Timeout` 形态），不再拖到首次发请求。
- **修复超出双精度范围的整数让校验层集体失效**：`float()` 与 `math.isfinite()` 对任意大的 Python int 都会先尝试转成 C double，超出 `DBL_MAX`（约 1.8e308）时抛 `OverflowError: int too large to convert to float`。该异常继承 `ArithmeticError` 而**不是** `ValueError`，因此穿透了项目里**全部**以 `ValueError` 为契约的守卫——`get_settings` 的三个 handler、`SessionConfig` 的 `__setitem__`、各 provider 的 options 校验、`EmbeddingService` 的 `except ValueError`、questionary 的 validator。输入完全可达：`config.toml` 的整数是无上界的 TOML 整数，`tomllib` 原样解析成 Python int，而 `options` 是文档指定的扩展入口——实测 `[rerank_configurations.x.options] timeout = <401 位十进制整数>` 让 `JinaProvider._request_timeout` 抛裸 `OverflowError`，配置里写一个超大整数同样让 `chat_temperature` 绕过脱敏直接打出 16 帧裸 traceback。更隐蔽的是，把 `math.isfinite(x)` 当守卫等于没有守卫：它自己就会抛 `OverflowError`。现在新增 `is_double_convertible`／`to_finite_float`／`safe_repr` 三个 helper，判据改为「先确认在双精度范围内，再转 `float`」，并覆盖 **21 处**收敛点：`Settings` 的 `chat_temperature`／`chat_vector_weight`／`chat_keyword_weight`／`chat_score_threshold`，`SessionConfig` 的权重、阈值与温度（构造器与 `__setitem__` 两条入口），Jina／SiliconFlow 的 `options.timeout` 与响应 `relevance_score`，`openai_compatible` 的后台轮询参数，Volcengine 的客户端 timeout 与文件等待参数，Google 的 `http_options.timeout` 与请求 timeout，Ark Responses 的 `fps`，`normalize_embedding_vector` 的向量项，`coerce_score`（展示层与检索层共用的分数收敛规则，契约是「绝不抛异常」），以及 `RetrievalService._validate_rerank_output`。
- 修复错误消息里的 `{value!r}` 反过来破坏异常：CPython 的 `sys.get_int_max_str_digits()`（默认 4300）让 `repr(10**4301)` 抛 `ValueError: Exceeds the limit (4300 digits)`，于是 `top_k` 的消息变成与字段无关的「整数转字符串超限」。新增 `safe_repr`（超长截断、超出位数限制时用 `bit_length()` 给出量级），`Settings` 与 `SessionConfig` 的校验消息统一改用它。
- **补上同一缺陷在裸插值站点的残留**：上面那条只覆盖了用 `!r` 的格式化，但**裸插值**（`{value}`）同样会在正要报告错误时改抛另一个错误。逐站点复查后又修了三处，都带红绿验证：`require_token_count` 的**负数**分支（`{numeric}` → `safe_repr`）——正数在 `numeric < 0` 之前就返回了，因此原有那条「永不抛非 `RuntimeError`」的用例（探针全是正数）碰不到它，实测 `-(10**4301)` 让契约里承诺的 `RuntimeError` 变成 `ValueError: Exceeds the limit`；`Settings.retrieval_candidate_multiplier` 的**下界**分支（正数在**上界**判断处被拒、消息里是常量，同样只有负数能走到）；以及 `RecursiveTextSplitter._validate_hierarchical_sizes` 的两条尺寸消息。另修两处同类但**判据不同**的缺陷：`SessionConfig._validate_fusion_strategy` 与 `RecursiveTextSplitter` 的 `structure_mode` 用集合成员判断（`value not in {...}`），对不可哈希的取值抛 `TypeError: unhashable type`——本类契约是 `ValueError`，调用方的 `except ValueError` 接不住；两者都补上显式判类型。回归用例刻意做成**全字段扫描**（59 个字段 × 正负两个方向，断言消息里绝不出现 `Exceeds the limit`）而不是逐字段补断言——这类站点散落在各 validator 里，逐个补挡不住下一个。
- **修复 `kb_chunk_overlap`／`kb_child_chunk_overlap` 缺上界**：两个字段原先只有下界，是 59 个字段里**唯一**能被超大整数静默通过的两个（实测全字段扫描）。这是**配置可达**的：`config.toml` 写 `kb_child_chunk_overlap = 10**400`（401 位，`tomllib` 照常解析）后配置加载成功，直到分片期才在 `_validate_hierarchical_sizes` 抛错——报错现场离写错的配置很远，消息里还会打出整个 401 位数字。上界取 `_MAX_KB_CHUNK_TOKENS` 与 `kb_chunk_size` 同源（overlap 的语义是「必须小于 chunk_size」，而 chunk_size 的上界就是它），不是另写一个数。另外 `RecursiveTextSplitter` 的构造器参数是 `Settings` 之外的第二条入口，上界拦不住它，因此那两条尺寸消息同时改用 `safe_repr`——4301 位的值经这条入口时，消息插值自己会抛，把「overlap 必须小于 size」换成与重叠参数无关的位数上限错误。
- **修复 Excel 召回测试日志静默丢失**：`log_results` 的 `workbook.save()` 外面是 `except Exception: logger.exception(...)`，因此**任何**写入期异常都让整个工作簿一个字都没落盘，用户只看到日志里一行「保存 Excel 日志文件失败」。而「形态完全合法的 int」正是这样的输入——两条下游路径的容忍度不同：`openpyxl` 写单元格前会调 `isnan(value)`，对超出双精度的整数（`10**400`，401 位）抛 `OverflowError: int too large to convert to float`；`json.dumps` 则对位数超限的整数（`10**4301`）抛 `ValueError: Exceeds the limit (4300 digits)`。这两个值来自第三方 store 的 `page`／`semantic_score`／`keyword_score`，而 `_cell_value`／`_json_safe` 的职责正是「收敛成下游能接受的形态」，却都用 `isinstance(value, int)` 直接放行。现在两个 helper 各自按下游的真实容忍度收敛（`openpyxl` 侧用 `is_double_convertible` 判双精度范围，`json` 侧判 `str()` 是否可转换），降级一律走 `safe_repr`——位数超限时连 `str()` 都会抛，用它等于把一类错误换成另一类。位数在上限内的整数照常保留原类型，日志的用途是记录真实取值。
- 顺带修掉这条路径上的一处**假绿**测试：`test_excel_log_survives_malformed_documents` 原判据是 `os.path.exists(logger.filepath)`，而 `save()` 失败时 `openpyxl` 已经写下了部分 ZIP 内容——文件存在且非空（实测 2290 字节），`load_workbook` 却抛 `KeyError: There is no item named '[Content_Types].xml' in the archive`。因此「文件存在」在未修复的代码上也成立，用例一直是绿的。判据改为读回工作簿并断言数据行存在。
- **修复 `safe_exception_text` 自己会抛异常**：它是**最后一道展示边界**，22 处 `except` 都经它输出——它再抛，用户看到的就是解释器默认 handler 打出的裸 traceback，连脱敏都没有，比「消息不好看」严重得多。触发条件是异常自身的 args 含位数超限的整数：`str(10**4301)` 撞上 CPython 的 `sys.get_int_max_str_digits()`（默认 4300）抛 `ValueError`。可达性经核实：JSON 与 TOML 两条路径送不进来（解析器自己就抛），但 **`pickle` 是二进制格式、不受十进制位数限制**——实测 `pickle.loads(pickle.dumps(10**4301))` 正常往返，而本项目加载 pickle 知识快照，因此快照里的畸形值可以携带这种整数进入异常对象。现在兜住 `str()`，降级时**逐个渲染 `exc.args`** 而不是把整个元组交给 `safe_repr`（后者会递归到元素上、兜住后只能退回 `<tuple>`，把「是哪个值出的问题」也丢掉）。`safe_repr` 走局部导入：`config` 在模块层导入本模块，模块层反向导入会成环。
- **修复 `VERSION.txt` 编码损坏让程序无法启动**：`_resolve_version()` 在**模块导入期**执行，早于 `main()` 的任何 `try`，异常会直接终止进程。它自带两级兜底（`VERSION.txt` → `pyproject.toml` → `"0.0.0"`），但原实现只覆盖了「文件不存在」——读取本身也会失败。实测 `VERSION.txt` 被写成 UTF-16 或被截断时 `read_text` 抛 `UnicodeDecodeError`，程序连横幅都打不出来就退出（exit code 1，消息是裸 traceback），这与兜底链的设计意图相反。现在读取一并兜住 `OSError` 与 `UnicodeDecodeError`。
- 修复测试自身的同类问题：`10**4301` 作为 pytest 参数值时，pytest 自动生成 id 会调 `str(val)` 并抛 `ValueError`，测试在**收集阶段**就崩掉、一条都跑不到。新增用例一律显式给出参数 id。
- **修复二进制发布包的核心功能不可用**：PyInstaller 的静态分析看不见 `tiktoken` 用 `pkgutil.iter_modules` 对 `tiktoken_ext` 命名空间包的运行时扫描，漏收 `tiktoken_ext.openai_public` 后 `get_encoding("cl100k_base")` 抛 `Unknown encoding`，知识库向量化在分片第一步即失败。`--smoke-test` 此前只打印版本号与路径，从不触达该路径，因此构建校验通过而产物是坏的。现把该模块加入 `HIDDEN_IMPORTS`，并让自检真实构造 `RecursiveTextSplitter` 并调用 `split()`。
- **修复二进制包离线首次运行失败**：`tiktoken` 在缓存未命中时会联网拉取编码文件。构建脚本现把 `cl100k_base` 的 BPE 预置进包内 `tiktoken_cache/`，并按 `sha256` 校验后写入，坏数据不会进包；缓存缺失时由构建脚本自行下载，不再要求构建者先手工跑一次知识库构建（此前 CI runner 上没有该缓存，五个发布目标会全部构建失败）。
- **修复预置 BPE 被环境变量旁路**：`main.py` 原用 `os.environ.setdefault("TIKTOKEN_CACHE_DIR", ...)`，而 `tiktoken` 的 `read_file_cached` 只看该变量**是否存在**，不存在「先查它、再回落包内」的语义。环境里一旦已有该变量（CI、容器、用户显式设置），包内文件就被完全旁路，离线运行退回联网并失败。现改为赋值，并在目录缺失时保持不设置。
- **修复只读安装目录下无法启动**：日志层与缓存目录两处都在模块导入期或 `main()` 的 `try` 之外无条件 `makedirs`，安装在 `/opt`、`/Applications`、`C:\Program Files` 这类只读位置时报裸 `PermissionError` 且无脱敏。现两处均降级：日志仅输出到控制台，缓存目录回落到系统临时目录。
- **修复二进制包内运行时目录位置错误**：`prepare_runtime_layout` 把 `knowledge_base/`、`data/kb`、`data/logs` 建在包根，而冻结时 `resolve_app_root()` 返回可执行文件所在目录（包内 `PyRAG-Kit/` 子目录），用户按 README 放入文档只会拿到 `FileNotFoundError`。现统一以该子目录为根，`config.toml.example` 与 `.env.example` 也一并移到那里（配置与 `.env` 都按该目录解析，放在包根会被静默忽略）。
- **修复 `.env` 从进程工作目录读取**：`env_file` 原为相对路径 `".env"`，由 pydantic-settings 按 CWD 解析，而 `config.toml` 走 `ROOT_DIR`。冻结产物中可执行文件目录与调用者 CWD 通常不同，用户放在安装目录旁的 `.env` 会被静默忽略。现两者同源，均以 `ROOT_DIR` 为基准。
- 修复构建脚本的 BPE 缓存查找与 `tiktoken` 语义不一致：原按变量真值判断，空串会落回临时目录，而 `tiktoken` 把空串视为「禁用缓存」并直接走网络。
- 修复混合检索权重可被设成同时为零：`Settings` 启动时校验过，但 `SessionConfig` 是运行期独立入口，单字段界拦不住这种组合。两个权重同时为零时 `weighted` 融合的每个文档得分都是 0.0，再被 `chat_score_threshold` 全部过滤，检索静默返回空结果且无报错。现在 `SessionConfig` 的设值入口与构造器都拒绝，`HybridReranker._weight_tuple` 另有兜底以覆盖绕过前两者的库调用方。
- 修复发布包版本号可能漂移：包名来自 `pyproject.toml`，而横幅与自检输出来自 `main.py` 的硬编码常量，只改一处会让两者不一致，而 `--validate` 只 grep `smoke test ok`、不带版本号，构建照样通过。现由构建脚本把版本写入包内 `VERSION.txt`，`main.py` 优先读取（按应用根目录查找，而非 `__file__`——冻结后后者指向 `_internal/`），并在 `validate_bundle` 中断言自报版本与 `pyproject.toml` 一致。
- 修复归档顺序与自检残留：原先先归档再校验，校验失败时归档已落盘且内容是校验前的状态；自检产生的空日志与 `.cache` 还会被一并打进发布包（目录版与归档版条目数因此不一致）。现改为校验 → 清理残留 → 归档。
- 发布工作流补上产物体检与校验和：此前归档只按名字 glob 上传，包内混进 `.env`、`config.toml` 或漏掉 `tiktoken_cache/` 都不会被发现，用户下载到的包可能既泄露凭证又无法离线运行。现在归档后显式拒绝这两类文件、要求缓存目录存在，并生成 `SHA256SUMS.txt` 一并发布。
- 修复发布工作流的归档校验步骤形同虚设：该步骤用 `release_artifacts/*.tar.gz` 单层 glob 查找归档，而 `download-artifact` 把每个 artifact 解到 `release_artifacts/<target>/` 下一层，glob 一个都匹配不到，循环体不执行、`fail` 保持 0，整步静默通过——校验逻辑写了等于没写。现改为递归查找（与 publish 步骤的 `maxdepth 2` 一致），并在归档数为 0 时直接失败，避免下载步骤出问题时以「没有可校验的东西」收场。同时把 `.zip` 与 `.tar.gz` 合并到一条分支，Windows 目标此前只在被匹配到的情况下才校验。
- 修复校验和的落盘位置与条目格式：`SHA256SUMS.txt` 原写在 `release_artifacts/` 内（正确），但条目保留了 `release_artifacts/<target>/` 前缀，用户单独下载某个归档后 `shasum -c` 找不到文件。现改为只写文件名，并把哈希工具从 `shasum`（依赖 perl）换成优先 `sha256sum`（coreutils 标配），无归档时显式失败。
- 发布流程补上测试门禁：`release.yml` 此前只构建、从不跑测试，打标签的提交若没走过 PR 门禁（例如直接 push tag 或 `workflow_dispatch`），产物会在测试全红的情况下被发出去。现新增独立的 `test` job 并让 `publish` 依赖它。刻意不塞进 `build` 矩阵——那是五平台，其中四个平台本仓库从未在 CI 上跑过测试，拿它们当发布门禁等于用一个没人验证过的信号去拦发布；这里固定在 ubuntu-24.04，与 `quality.yml` 同一环境。
- 包内缺少分词编码缓存时显式警告：`tiktoken` 在缓存未命中时走 `requests.get(blobpath)`，而该调用**没有超时**（见 `tiktoken/load.py` 的 `read_file`），网络黑洞下会长时间挂起而非快速失败。包内 `tiktoken_cache/` 是唯一的离线兜底，缺失时现在向 stderr 输出提示，用户至少知道卡在哪。

- **修复退出钩子可能删除整个应用目录**：`cache_path` 是用户可配置项，而 `cleanup_temp_files()` 在 atexit 里对其执行 `shutil.rmtree`。原有保护只挡 `/`、家目录与一份硬编码的 POSIX 系统目录清单，拦不住指向应用自身的取值：实测冻结产物中把 `cache_path` 配成 `"."`（解析为应用根目录）后正常退出，整个安装目录被删光——`.env`（内含 API Key）、`knowledge_base/`（用户放入的源文档）与 `config.toml` 一并消失且无法恢复。`"data"`、`"knowledge_base"`、`"data/logs"`、`"data/kb"` 同样会被删掉，分别波及知识快照、源文档与日志。现在判定改为与应用真实使用的路径（应用根目录、`config.toml`、`.env`、`knowledge_base_path`、`snapshot_root`、`log_path`）比对，相等或位于其下即拒绝，不再依赖目录名。
- **修复日志清理会删除其他程序的日志**：`cleanup_old_logs()` 原匹配「目录下所有 `*.log` 且文件名含任意 `YYYY-MM-DD`」，而 `log_path` 同样是可配置项。指向共享目录（`/var/log`、挂载的日志盘）时，退出前的清理会连带删掉 `nginx_access_2020-01-01.log`、`postgres_2020-01-01.log` 乃至 `backup_2020-01-01.log` 这类与本应用无关的文件，且不可逆。现在要求文件名严格等于本应用的命名形态（`app_log_<日期>.log` / `chat_log_<日期>.log`），前缀由写入与清理共用的常量定义，避免两侧规则各自演化。

- 修复终端展示 `base_url` 时不脱敏：自建网关常把凭证写在 URL 里（`https://user:token@gateway/v1`），`api_key` 一直有掩码，但 Base URL 是原样打印的——日志侧有 `RedactingFormatter` 兜底，终端这一路没有，凭证会直接出现在屏幕上、终端回滚与用户的截图里。现在显示前先经 `redact_sensitive_text`，主机名仍可见以便确认连的是哪个网关。

- 活动快照标记文件的内容不再直接当路径分量使用：`get_active_snapshot_id()` 读取 `ACTIVE_SNAPSHOT` 后把它拼进 `self.root / snapshot_id`，而该文件是磁盘上的任意文本——绝对路径会**顶掉** root（`Path("/a") / "/b"` 得 `/b`），`..` 与 `a/../..` 则向上逃逸，都让「活动快照」指到快照根目录之外。当前两个调用方之后都会再走 `resolve_within`／`validate_snapshot_dir` 复核，尚不构成实际逃逸，但本方法的对外语义就是「root 之下的活动快照」。现在要求解析后必须直接位于 root 之下，判定用解析结果而非字符串是否含分隔符（后者漏掉不含分隔符却能跳出 root 的 `..`）。

- 快照 id 的路径校验覆盖到全部拼路径的方法：上一轮只收紧了 `get_active_snapshot_id()`，而 `create_temp_snapshot_dir`、`cleanup_temp_snapshot_dir`、`finalize_snapshot` 同样把 `snapshot_id` 拼成 `self.root / snapshot_id` 并对其执行 `rmtree`——实测 `finalize_snapshot(root/".tmp-x", "../../<victim>")` 会把快照根目录**之外**的目录整个删掉，而当时的入口校验拦不住（调用方传参不经过标记文件那条路）。校验现在集中到一个 helper，四个方法共用，不再各判各的。

- 修复混合检索在缺少分片标识时会把两篇不同文档并成一条：`_document_key` 的兜底分支取 `page_content[:64]` 作键，同来源、前 64 字符相同的两篇文档因此碰撞——实测 `_merge_hybrid_results` 把语义结果里的那一篇覆盖掉、只返回一条，用户少拿一篇且没有任何提示（`_deduplicate_parent_documents` 同理，丢的是得分较低的那篇）。现改为对整段内容取 sha256：既保持定长，又不因截断而碰撞。`chunk_id` 在真实快照里恒存在，这条是兜底路径，但兜底同样不能产生错误结果。

- 修复 `SessionConfig(...)` 构造器绕过逐字段校验：这是**整类**绕过而非单个字段——实测修复前构造器对 `top_k=10**9`、`top_k=True`、`score_threshold=nan`、`chat_temperature=-1`、`hybrid_fusion_strategy="bogus"`、`rerank_enabled="yes"` 等每一个受校验字段都照单全收，而 `__setitem__` 会逐一拒绝。其中 `top_k=10**9` 会一路走到 `faiss_index.search()`，FAISS 既不报错也不截断，按 10 亿条分配内存；`nan` 权重则让 `HybridReranker` 算出 `nan` 得分，排序结果不确定且与阈值的比较恒为 False，整批结果被静默丢弃。构造器现在与 `__setitem__` 共用同一组校验器，两条入口对同一取值的结论一致。
- 修复 `HybridReranker._weight_tuple` 对非有限权重的兜底：原判据 `total <= 0` 对 `nan` 恒为 False（`nan <= 0` 不成立），`vector_weight=nan` 会一路算出 `nan` 得分。现在同时要求和为正的**有限**数值。这是所有构造路径的最后一道兜底，覆盖绕过 `Settings` 与 `SessionConfig` 的库调用方。

- 修复候选规模的上界没有落在乘积上：真正送进 `faiss_index.search()` 的是 `effective_top_k = chat_top_k × retrieval_candidate_multiplier`，而两个字段此前只有各自的上界（10_000 × 100 = 10⁶）。FAISS 对超过索引条数的 k 既不报错也不截断，按 k 全量分配——实测 dim=1536、k=10⁶ 时峰值 RSS 涨约 35 GB。现在 `Settings` 与 `SessionConfig` 都按乘积判定，单字段上界同步收窄到 2_000 使两个约束不再互相矛盾（此前 `top_k=10_000` 能通过字段校验却被组合校验拒绝）。
- 修复配置编辑器放行「字段合法、组合越界」的取值：UI 的两个校验器只按单字段上界判断，用户把 `top_k` 设成上界值（UI 放行）后，应用配置时 `__setitem__` 抛 `ValueError`——而该异常落在 chat 主循环之外（主循环只捕获 `KeyboardInterrupt`/`EOFError`，外层 `try` 只有 `finally`），会以裸 traceback 终止整个会话。现在 UI 的上界按当前另一字段的取值动态计算，放行的值必定能被 `SessionConfig` 接受。

- 修复配置批量更新不是原子的：`Chatbot.chat_config` 的 setter 收 dict 时逐个写 `session_config`，任一字段被拒时前面的字段已经落进去了——实测 `{"top_k": 42, "retrieval_method": "不存在"}` 会留下「`top_k` 已改成 42、检索模式还是旧的」这种改了一半的状态，而调用方只看到一个异常，以为整批都没生效。现在先在副本上写完再整体替换。

- 修复快照校验漏掉两个索引文件的行数：`validate_snapshot_dir` 的文档说明写着要把各数据源「对齐」，但实际只比对了 `chunks.pkl`、`embeddings.npy` 与 `stats.json`，**没有检查 `semantic.index` 与 `lexical.index`**——而加载路径两者都用盘上的文件。实测 `semantic.index` 的 `ntotal=5` 而 `chunks.pkl` 为 3 条时校验放行，加载后 `semantic_search` 拿 `indices` 去索引 `self.documents` 抛 `IndexError`；`lexical.index` 语料 5 条而分块 3 条时同样放行，`keyword_search` 在命中越界位置时抛 `IndexError`。现在两个索引文件的行数都纳入校验。

- 修复基础清洗器会改写用户内容：`BasicCleaner` 把双换行临时替换成硬编码的 `"__DOUBLE_NEWLINE_PLACEHOLDER__"`，与用户文档里同名字面量碰撞。实测 `"本文档说明 __DOUBLE_NEWLINE_PLACEHOLDER__ 的用途。"` 清洗后变成 `"本文档说明 \n\n 的用途。"`（原文被改写），而整段由该串构成的文档清洗后得到空串（内容全部丢失）。哨兵现在按「内容里不存在的字符」现算（NUL，存在时继续追加），不再是可与正文碰撞的固定字面量。
- 修复向量写入在维度校验前就已改动存储：`upsert_embeddings` 原先先 `documents.extend(...)` 再比对维度，维度不一致时抛异常但 `documents` 已经多出一批没有对应向量的条目。实测拒绝之后 `documents=2` 而 `embeddings` 仍为 `(1, 4)`，store 已不自洽，而 `save_snapshot` 会照常把这份不自洽落盘——之后的语义检索拿 `indices` 去索引 `documents` 会越界或错配。校验现在前置，失败不留痕。

- **修复终端输出把不可信文本当作富文本标记解析**：`console.print(str)`、`Table.add_row(str)`、`Panel(str)` 在 rich 里默认按**标记**解析，而本项目的这些位置里插着知识库文档正文、用户输入、配置值与异常消息。后果有两类。一是崩溃：正文里出现字面量 `[/]`、`[/color]`、`[/if]` 时抛 `MarkupError`，实测知识库文档里「匹配斜杠用 [/] 表示」这一行即可触发；真正致命的是**异常处理走的是同一个通道**——`safe_exception_text()` 拼出的消息里带着 `[/]`，于是召回测试的内层 `except` 抛、`main()` 的兜底 `except` 也抛、`run_cli` 的兜底 `except` 再抛，三层兜底全部失效，程序以裸 traceback 退出（已端到端复现）。二是静默改写：`[b]粗体[/b]` 的标记被吃掉、markdown 链接 `[https://x](https://x)` 丢掉方括号部分，实测真实知识库 3038 行里有 12 行渲染后被改写，用户看到的与检索到的不一致。现在新增 `display_utils.markup_text(template, *values)`：**模板**继续走标记解析（样式照旧生效），**值**通过 `Text.append` 原样追加并继承所在位置的样式；聊天、召回测试、知识库构建、配置编辑与两个顶层错误出口共 30 处调用点全部改走它。`display_chat_config` 的 Base URL 单元格另有同类问题：该值来自 `config.toml`／`.env` 且未经校验，IPv6 字面量本身带方括号，现改用 `Text` 包装（脱敏解决凭证泄漏，不解决标记解析）。
- 修复快照校验漏掉 `parents.pkl` 的解析与形态：`validate_snapshot_dir` 已把 `chunks.pkl` 与 `lexical.index` 的解析失败转成「快照不自洽」的 ValueError，却漏了同批 `pickle.load` 的 `parents.pkl`。实测它写坏时校验照常通过，直到 `FaissStore.load_snapshot` 抛裸的 `UnpicklingError: invalid load key, 'n'.`；顶层写成 `None`／`int` 则在 `dict(self.parent_documents)` 处抛 `TypeError: 'NoneType' object is not iterable`；父条目不是 dict／str 时在 `_normalize_loaded_documents` 抛 `'list' object has no attribute 'get'`。三者都没有「快照坏了」的语义与重建指引，而同一份快照里 `chunks.pkl` 坏掉时给出的正是有指引的 ValueError——同类失败不该有两种收场方式。现在三处都在加载前拒绝。

- 修复召回测试的 Excel 日志把文档内容当公式写入：`openpyxl` 按值猜类型，以 `=` 开头的字符串会被存成**公式**（`data_type='f'`），Excel 打开日志时求值。文档内容、来源与查询都来自知识库与用户输入，实测 `=1+1` 落盘后是公式而非文本，`=cmd|'/c calc'!A0` 这类 DDE 载荷同样照单全收。两条后果：日志不再忠实记录「检索到了什么」（内容列显示的是计算结果），以及在打开日志的机器上执行文档里的内容。现在字符串单元格显式写 `data_type = "s"` 固定为文本——用类型而不是加 `'` 前缀，因为前缀会改写内容，而日志的用途正是逐字记录。

- 修复 manifest 的 TOML 序列化不做转义：`to_toml` 用 `f'key = "{value}"'` 直接拼字符串，而这些值都来自 `config.toml`／`.env`——`embedding_model` 可以是自建网关的模型路径（`models\qwen-7b`），`chunk_mode` 与 `snapshot_id` 也由调用方决定。实测模型名含反斜杠时 `write_manifest` 正常落盘、`load_manifest` 抛 `TOMLDecodeError: Unescaped '\' in a string`：快照**写得出、读不回**，报错现场离真正的原因很远；含引号或换行的值则连落盘的 TOML 都不可解析。同一份 manifest 在内存里是好的，落盘一圈就坏了。现在按 TOML 1.0 转义 `"`、`\` 与不可打印字符（可打印字符含中文原样保留），计数字段显式 `int()` 以免被写成带引号的字符串。

- 修复 `config.toml` 语法错时绕过脱敏边界：`get_settings` 的 `try` 只接 `ValidationError` 与 `SettingsError`，两者都覆盖不到 `tomllib.TOMLDecodeError`——它继承 `ValueError`，但既不是前者也不是后者（pydantic-settings 在 `load_toml_config` 里把 tomllib 的异常原样放出来）。而 `get_settings` 在 import 期就会被调用（`log_manager.get_module_logger`），早于任何入口的 `try`：实测 config.toml 语法错时 import 直接抛裸 traceback，**35 帧全打在终端上**；作为对照，同一份文件里字段**取值**非法时（`chat_top_k = "abc"`）得到的是 11 帧的脱敏消息。同属「配置写错」，收场方式不该有两种。现在两条路径都产出 `配置解析失败 - ...` 的一行消息，并保留行号列号供定位。

- 修复 embedding 维度没有上界：`local-hash` 的维度直接来自用户可写的 `model_name` 数字后缀（`local-hash-256` -> 256），而 provider 会**先按该维度分配出完整向量**再返回。实测配成 `local-hash-100000000` 时 `Settings` 照单全收，单次 `embed_texts(["测试"])` 耗时 136 秒、峰值内存 3.9 GB；按真实知识库 115 个分块算，落盘前的 `np.vstack` 需要约 4.3 GB，进程会被 OOM 杀掉而不是得到一条可读的错误。这与已经加过界的 `chat_top_k × retrieval_candidate_multiplier` 是同一类问题——用户可配的规模没有上界。现在上界为 65536（现有各家的最大维度是 3072，留 20 倍余量），在 `local_hash` 的**构造期**就拒绝，因此没有任何分配发生过（实测同一配置从 146 秒 / 3.9 GB 降到 6.8 秒 / 4 MB）；`EmbeddingService` 另在 `np.array` 之前按同一常量兜底，覆盖其它 provider 返回超大维度的情况。

- 修复载入快照时未校验文档是否合规：`Document` 的契约是 `content: str` + `metadata: dict`（见 `src/models/document.py`），但载入路径把快照／legacy pkl 里的条目直接当 `Document` 用，只做 `metadata or {}` 的兜底。实测 `page_content=None` 的快照能通过 `validate_snapshot_dir`、加载成功，直到 `display_results` 里 `.replace()` 才抛 `AttributeError: 'NoneType' object has no attribute 'replace'`——报错现场离真正的原因（快照里这一条不是字符串）很远；条目本身不是映射、`metadata` 非映射同理。现在在 `FaissStore` 的载入边界拒绝，理由与 `chunks.pkl` 的解析检查一致：这是快照内容不自洽，不是显示层的问题。`page_content` **缺失**仍放行（各消费者都写了默认值），`None` 则拒绝——它在 `Document` 里是必填的 `str`。
- 修复展示与 prompt 构造被形态异常的检索结果打断：`display_results`（召回测试）、`Chatbot._display_retrieved_docs`（聊天）与 `ChatService._build_prompt` 都直接对结果做 `.get(...).replace(...)` 与 `f"{score:.4f}"`。检索结果可以来自任何 `VectorStoreBase` 实现——`FaissStore` 会在载入期校验并归一化，**第三方 store 不会**，而抽象基类正是为替换实现而存在的。实测一个返回 `{"page_content": None, "metadata": None, "score": None}` 的 store 能让这三处分别抛 `AttributeError` 与 `TypeError`；更隐蔽的是 `page_content=None` 在 prompt 路径上**不抛错**，而是把字面量 `内容: None` 当作一条真实知识喂给模型。现在 `display_utils` 新增 `document_content` / `document_source` / `document_score` 三个提取助手统一收敛形态（`bool` 与 `nan` 也退回默认值——`True` 会显示成 `1.0000`，看起来像真实得分），`RetrievalService` 另在 `retrieve` 出口校验「文档映射的列表」这一接口契约，`metadata` 非映射则归一化为空 dict 后继续（元数据异常不该让整批检索失败）。

- 修复 `markup_text` 静默丢弃样式：该助手原用 `Text.from_markup(...).split(占位符)` 切段、再从**切出来的相邻段**上取样式，而占位符本身正是被切掉的那一段——标记解析出的样式恰恰只挂在它身上（`[bold cyan]\x00[/bold cyan]` 的 span 是 `(0,1,bold cyan)`），切段后右侧是空串、没有样式。于是「值独占标记内容」这一**最常用**的写法（仓库里 13 处 `[bold magenta]{}[/bold magenta]` 都是这个形状）渲染出来完全没有样式，且不报错。反向的错位同样存在：靠「已拼接结果末字符」兜底会让 `[bold]x[/bold]{}` 把值染成粗体，而值在标记**之外**。现在按占位符的位置把整段解析结果切开重组，样式取自占位符自身覆盖的 span；实测 22 种模板形态与 `escape` 参考逐字符一致，真实知识库 4489 行 0 改写、0 崩溃、0 样式不符。

- 修复配置面板把 `config.toml` 派生值当标记解析：上一轮只把「融合策略」与 Base URL 两个单元格包成了 `Text`，其余仍是裸 f-string／`str()`。这些取值同样没有白名单——provider 工厂只校验 `provider`（`_provider_map` 白名单），`model_name` 与配置**键名**（`active_llm_configuration`／`active_rerank_configuration`）原样透传。实测 `model_name = "gemini[/]"` 时 `reload_llm()` 返回 True（模型确实建起来了），紧接着的 `display_chat_config` 抛 `MarkupError`——聊天会话在启动横幅这一步就崩，用户连配置都看不到；`active_*_configuration` 是配置表的键名，任意字符串即可，同样工厂放行、面板崩溃。现在整张表的单元格与行标签统一按字面呈现。构建脚本 `embed_knowledge_base.py` 的配置总览表有同类问题，且它在**任何校验之前**运行（`main_async` 先调它、再建服务），`provider` 里的 `[/]` 会让「是否使用以上配置继续？」这一步直接崩溃；该函数此前没有任何测试覆盖。

- 新增结构性守卫，把「rich 解析不可信值」这一类问题钉死：此前每个调用点各有一个回归测试，但这类问题的共同点是「新写的代码又把裸字符串交给了 rich」，逐个补测试挡不住下一次。现在用 AST 检查 `main.py`／`src/`／`scripts/` 的全部源码，凡是会解析标记的入口（`console.print`、`Table.add_row`、`Panel`）其参数必须是字面量或已由 `Text(...)`／`markup_text(...)` 包装。确属外部输入不可达的少数位置逐条登记理由，并有单独的测试断言登记表没有失效条目（行号漂移即报错）。

- 修复快照校验漏掉 `manifest.toml` 的解析：`validate_snapshot_dir` 把 `chunks.pkl`／`parents.pkl`／`stats.json`／`semantic.index`／`lexical.index` 的解析失败都转成了带重建指引的「快照不自洽」ValueError，唯独 manifest 只查文件**是否存在**、不查能否解析。而 `VectorStoreFactory` 启动时正是「先 `validate_snapshot_dir`、再 `load_manifest`」，实测把 manifest 写成 `a = "unterminated` 后校验照常通过、紧接着抛裸的 `TOMLDecodeError: Unterminated string`——用户看到的是纯语法描述，既没有「快照坏了」的语义也没有重建指引。现在两个入口都按同一方式收场。

- 修复融合链路不收敛分数字段的形态：展示层已有 `document_score` 收敛（`bool`／`nan`／非数字退回默认值），但 `_merge_hybrid_results`、`HybridReranker`、`_deduplicate_parent_documents` 与阈值过滤都直接 `float(doc.get("score") or 0)`。检索结果可以来自任何 `VectorStoreBase` 实现——`FaissStore` 归一化，第三方 store 不会。实测两种失败方式：`score="abc"` 抛 `ValueError`、`score=[1]` 抛 `TypeError`，异常从融合函数一路冒到 chat 主循环之外；更隐蔽的是 `nan`——`_normalize_scores` 里 `max()`/`min()` 遇到它双双返回 `nan`，`score_range` 也是 `nan`，每个文档的归一化结果都成了 `nan`，而 `sorted` 对 `nan` 的比较恒为 False，该文档保持原相对顺序**留在首位**：一条分数不可用的结果被排到正常结果之前，且没有任何提示。`inf` 同理，`max()` 取到它之后其余文档的归一化结果全被压成 0。现在新增 `_finite_score` 统一收敛（要求有限性，`bool` 也退——`True` 会算成 `1.0`，看起来像真实得分），七处取值点共用；融合产出的 `score` 字段本身也一并收敛，不让中间产物带着不可用的值往下游走。

- 修复 Excel 日志不收敛检索结果的形态：`log_results` 直接对每个条目做 `doc.get(...)`／`metadata.get(...)`，并把取值原样交给 `openpyxl` 与 `json.dumps`。检索结果可以来自任何 `VectorStoreBase` 实现，第三方 store 不经 `FaissStore` 的载入期归一化，实测六类形态抛异常：`metadata` 为 `None`／列表与条目本身不是映射都 `AttributeError`；`page` 是映射／列表时 `openpyxl` 抛 `ValueError: Cannot convert ... to Excel`；`semantic_score` 是集合时 `json.dumps` 抛 `TypeError: Object of type set is not JSON serializable`；`score` 是映射／列表同样进不了 Excel。日志是**旁路**——它失败不该让一次成功的召回测试以异常收场，也不该带走同一批里其余正常条目的记录。现在与展示层共用 `document_content`／`document_source`／`document_score`，并对单元格与 JSON 值各加一层形态收敛。

- 统一分数收敛规则到 `display_utils.coerce_score` 一处：上一项修完发现展示层与检索层各写了一份判据，且**结论不一致**——展示层拒绝数字字符串、检索层接受它。后果是同一份结果出现自相矛盾的解释：一个返回 `score="0.8"` 的第三方 store，其文档会被检索层按 0.8 通过 0.5 的分数阈值、却由界面显示成 `0.0000`，即「0.0000 排在最前面」，而它恰恰是通过了阈值的那些。现在两处共用同一实现，并明确三条规则与理由：数字字符串照旧转换（旧实现 `float(x or 0)` 接受它，一律拒绝会让字符串得分的 store 被静默过滤成空结果）、`bool` 与 `nan`／`inf` 退回默认值（`True` 会显示成 `1.0000` 像真实得分；`nan` 会让 `sorted` 的比较恒为 False 而把坏分数留在首位）。同时 `retrieve` 出口统一收敛 `score`：单路检索不经 `HybridReranker`，不会重算得分，第三方 store 的原始形态会一直流到调用方。

- 修复 legacy 导入的清单统计不收敛 `metadata` 形态：`factory.py` 的 legacy 分支用 `{doc.get("metadata", {}).get("source") for doc in store.documents}` 统计来源数，而该分支正是为**第三方 store** 准备的（`get_default_vector_store` 可以返回任何 `VectorStoreBase` 实现）。`FaissStore` 会在载入期归一化，第三方 store 不会——`metadata=None`／列表、甚至条目本身不是映射都会让这行抛 `AttributeError`，且抛在 `store.load()` 之后、`create_temp_snapshot_dir` 之内，正好落在需要 `finally` 清理的窗口里。现在与展示层共用 `document_source` 收敛。

- 修复批量配置更新按**旧值**做组合校验：`Chatbot.chat_config` 的 setter 在副本上逐个写字段，而每个字段的组合校验读的是**尚未被这批更新覆盖的旧值**。于是同一个合法终态能否被接受取决于 dict 遍历顺序——初始 `(top_k=5, mult=3)`、目标 `(top_k=2000, mult=1)`（乘积 2000 完全合法），先写 `top_k` 会拿旧的 `mult=3` 算出 6000 而拒绝，先写 `mult` 则通过。这不是理论问题：配置编辑器的动态上界会**放行**这两个字段（先设 `mult=1` 时上界 100，再设 `top_k=2000` 时上界正好 2000），用户以为设置成功，而应用配置时抛 `ValueError`——该异常落在 chat 主循环之外，会以裸 traceback 终止会话；目前只因 `to_dict()` 的键顺序恰好把 `retrieval_candidate_multiplier` 排在 `top_k` 之前才没暴露，嵌入方传入自己的 dict 或将来调整字段顺序就会踩到。现在 `SessionConfig` 新增 `set_many()`：先逐字段校验收敛到本地，再按「这批更新全部生效之后」的取值做组合校验，最后整体落盘；被拒时不留改了一半的状态。等值约束（两权重同时为零）保持原样——最后一次写入必然能看到另一个字段的新值，因此那条不受影响。

- **修复退出清理可能删除 legacy 知识库文件**：`_protected_app_paths()` 的保护清单漏了 `pkl_path`（legacy 知识库文件，用户投入的原始数据）。把 `cache_path` 指向 `data`、同时让 `knowledge_base_path`／`snapshot_root`／`log_path` 都落在 `data` 之外时，`data` 不覆盖任何受保护路径，`_reject_unsafe_cache_dir` 放行，退出清理随即 `rmtree("data")` —— 连带删掉 `data/employee_kb.pkl` 且不可恢复（已复现）。它与 `knowledge_base_path` 同属用户数据，现一并保护。
- **修复快照 id 的路径校验可被路径别名绕过**：判定原为「解析后的父目录等于 root」，它拦不住 `"x/../victim"`——该值解析成 `root/victim`，父目录**正是** root，于是放行。而三个方法实际操作的路径是 `self.root / f".tmp-{snapshot_id}"`，前缀 `.tmp-` 会**改变** `..` 的消解结果（`.tmp-x` 是一个真实路径分量），`".tmp-x/../victim"` 最终落在 `root/victim`，即一个**正式快照**：实测 `cleanup_temp_snapshot_dir("x/../victim")` 对 `root/victim` 执行了 rmtree，「只清理任务临时目录」的意图被绕过。现改为「必须是单一目录名」（不含分隔符、不是 `.`／`..`、非绝对路径），`generate_snapshot_id()` 的产物不受影响。
- 修复快照校验漏掉**向量维度**一致性：只比行数是不够的——实测分块数／embedding 行数／`ntotal` 都是 1，而 `embeddings.npy` 是 4 维、`semantic.index.d` 是 3 时校验放行，加载后 `semantic_search` 把 4 维查询向量送进 3 维索引，FAISS 抛的是**空消息的 `AssertionError`**（没有「快照坏了」的语义，也没有重建指引）。现在两个维度必须一致。
- 修复快照校验把 `chunks.pkl = None` 当作 0 个分块放行：加载后 `_normalize_loaded_documents` 对 `None` 迭代，抛裸的 `TypeError: 'NoneType' object is not iterable`。这是逐项校验（`_validate_loaded_document`）之前没检查**容器本身**的缺口，现在顶层必须是序列。
- 修复 hybrid 检索在校验前就合并两路结果：`_require_document_list` 原先只在合并与 rerank **之后**运行，而 `_merge_hybrid_results` 已经对每个条目调了 `.get(...)`——store 返回 `[None]` 时抛的是 `AttributeError: 'NoneType' object has no attribute 'get'`，而不是该校验承诺的可定位 `TypeError`。三条检索路径的收场方式因此不一致（单路给可读错误、hybrid 给裸 AttributeError），现在两路结果在合并前各自校验，错误消息指明是哪一路。
- 修复子分片尺寸关系在**非层级模式**下也生效：`kb_child_chunk_size < kb_chunk_size` 与 `kb_child_chunk_overlap < kb_child_chunk_size` 两条约束原先写在 `Settings` 的 model_validator 里，但**模式不在 `Settings` 里**——它是分片时的运行时参数（`build(splitter_structure_mode)`／CLI 的 `--mode`）。后果是一组「standard 下完全合法」的取值连配置都加载不了：实测 `kb_chunk_size=100, kb_child_chunk_size=300` 被 `ValidationError` 拒绝，而 standard 模式只调用普通分片器，子分片器的参数根本不参与。判定移到 `RecursiveTextSplitter._validate_hierarchical_sizes()`：层级分支一定会构造子分片器，非层级分支一定不会。
- 修复 `markup_text` 把转义的 `\[` 误判为标记：rich 的 `_parse` 按「前导反斜杠的奇偶」决定 `[` 是转义字面量还是真标记，而助手的守卫只看正则是否匹配到 `[`，于是 `r"\[example {}]"` 这类**合法**模板被拒绝，而 rich 自己会把它解析成 `[example <值>]`。现在按同一奇偶规则判定；`\\[`（双反斜杠）里的 `[` 仍是真标记，照旧拒绝。
- 修复一条 prompt 测试的假绿：`test_build_prompt_survives_malformed_documents` 只断言 `"问题" in prompt`，而该字符串在「没有找到相关知识」那一支里也有。实测把 `_build_prompt` 换成一个「丢弃全部知识、只回显问题」的实现，这条用例照样通过——它能发现崩溃，却**发现不了知识正文被静默丢弃**（那正是本项目最该防的失败：检索退化被伪装成模型答不出来）。现在对「有可用正文」的输入额外断言正文确实进入了 prompt。

- 修复**无键名的固定前缀凭证漏检**：SDK 的异常消息常回显请求头或 URL（`safe_exception_text` 的 docstring 写明了这一点），此时凭证是**裸串**，没有键名可锚定。既有的 `_TEXT_CREDENTIAL_KEYS` 覆盖的是键名——`volc_access_key=...` 会被脱敏，但 `invalid access key 'AKLT...'` 这种回显形态不会，实测 `RedactingFormatter` 对它原样输出。火山引擎（Ark）与阿里云都用这种固定前缀的长串凭证。现新增 `_AK_SK_PREFIXED_KEY_RE`（`AKLT`／`LTAI` 前缀 + 12 位以上），与既有的 `AIza`（Google）规则同类：前缀可自证，因此不依赖上下文。误伤评估：对真实知识库 4489 行、全部源码与文档扫描零命中，且 `AKLT 是前缀` 这类短串不受影响。

- 修复 Rerank 入参未收敛成字符串：`docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]` 在内容非字符串时会把 `None`／`42` 这类值直接交给 provider，由 provider 的 `_validate_inputs` 抛「documents 必须全部是字符串」——错误信息指向 provider 的参数，而真正的原因是上游 store 返回了形态异常的文档。现在与展示层共用 `document_content` 收敛，错误离原因更近。

- 修复整数解析在两处判据上都不完整：`str.isdigit()` 对 128 个 Unicode 码位为真（`²` `³` `①` `፩` `₅` …）却让 `int()` 抛 `ValueError`；即便换成 `isdecimal()`，**位数超过 `sys.get_int_max_str_digits()`（CPython 默认 4300 位）的数字串**`int()` 照样抛错——`"9" * 4301` 是合法的十进制整数。只堵住前一类，同样的崩溃只是换了个触发条件。现抽出 `parse_decimal_int`（`src/utils/config.py`）：`isdecimal()` 判形状 + `try/except ValueError` 兜住位数上限，契约是「解析不了返回 `None`，永不抛异常」，五处调用点统一改走它。受影响的三类现场各不相同：配置编辑器里输入 `²` 或粘贴一长串数字后回车，questionary 只把 `ValidationError` 当「输入无效」，其余异常会一路冒到 prompt_toolkit 事件循环之外，用户看到 `Unhandled exception in event loop` 加一段 traceback 且输入框卡死（实测经真实 `PromptSession` + 管道输入复现，异常自 `prompt.py::_accept_input` → `buffer.validate_and_handle` 抛出）；`local-hash-²` 这类 `model_name` 让构造期抛出一条只有 `invalid literal for int() with base 10` 的裸解析错误，既不提 `model_name` 也不提配置；`is_retryable_error` 与流错误事件提取则让**错误分类本身**崩掉——本该重试的 5xx、本该报出的流错误都变成解析异常；`_chat_status_is_error` 同理，会让「HTTP 200 但响应体写着失败」的兼容网关被误报成空文本。
- 修复 `config.toml` 里位数超限的整数逃出脱敏边界：`tomllib` 解析整数走的是 `int()`，位数超过 `sys.get_int_max_str_digits()` 时让裸 `ValueError` 穿出来——它**不是** `TOMLDecodeError`，`ValidationError`／`SettingsError`／`TOMLDecodeError` 三个 handler 全接不住。实测`chat_top_k = "9" * 4301` 得到 36 帧裸 traceback，消息只有 `Exceeds the limit (4300 digits)...`，既没有「配置解析失败」前缀也没说是哪个文件；而同属「配置写错」的语法错是 5 帧带前缀的脱敏消息。现补一个 `except ValueError` 兜底（排在三个具体分支之后——它们都继承 `ValueError`，顺序颠倒会让前面的分支永远接不到），五类配置写错（语法错、超长整数、字段取值非法、未知字段、坏分隔符）现在收场一致。

- 修复快照元数据的**字段取值**解析逃出「快照不自洽」的收场约定：`load_manifest` 与`validate_snapshot_dir` 此前只包住了 `tomllib.load`／`pickle.load` 的失败，解析**之后**的字段取值没包。`KnowledgeSnapshotManifest.from_mapping` 直接 `int(data[...])`／`data[...]`，于是 `chunk_count = "abc"` 抛裸 `ValueError: invalid literal for int()`、位数超限的数字串抛裸 `ValueError: Exceeds the limit (4300 digits)`、字段缺失抛裸 `KeyError`——`KeyError` 甚至**不继承** `ValueError`，连启动路径上的 `except ValueError` 都接不住。`stats.json` 的 `int(declared)` 同理（`"abc"`／列表／超长数字串分别抛 `ValueError`／`TypeError`／`ValueError`）。这些都属于「快照写坏了」，却都没有「快照不自洽」的语义与重建指引，而同一份文件语法错时给的正是那种消息。现两处都包住解析后的取值，五类形态（非数值字符串、浮点字符串、列表、位数超限的数字串、字段缺失）统一收场为带重建指引的 `ValueError`，并保留原始异常类型便于定位。经真实启动入口（`VectorStoreFactory.get_default_vector_store`）逐类复现确认。

- 修复 `save_snapshot` 在 `metadata` 为 `None` 时崩溃：`doc.get("metadata", {})` 的默认值只在**键缺失**时生效，键存在而值为 `None` 时它返回 `None`，随后的 `.get` 抛 `AttributeError: 'NoneType' object has no attribute 'get'`。同文件另外三处（`_build_index_text`、`_normalize_loaded_documents`、`_normalize_parent_document`）用的都是 `or {}`，只有 stats 计算这一处不一致。实测 `upsert_embeddings` 接受 `metadata=None` 的条目并正常建好索引（`_rebuild_indices` 走的是 `or {}`），到知识库构建的最后一步 `save_snapshot` 才崩——同一份数据两条路径的判据不一致。
- 修复 `Anthropic options.max_tokens` 的有限性判据晚于 `int()`：`int(float("inf"))` 抛 `OverflowError: cannot convert float infinity to integer`，`int(float("nan"))` 抛 `ValueError: cannot convert float NaN to integer`——两者都不是该方法承诺的「必须是正整数」消息，而 `OverflowError` 连 `except ValueError` 都接不住。实测 `max_tokens = inf` 正是这种收场，而同为非法取值的 `1.5`／`0`／`-1`／`True` 都能得到带指引的消息。现把有限性判据提到 `int()` 之前。
- 修复八个 token 计数方法的响应解析只判了 `None`：Anthropic 同步/异步 × 普通/Beta、Google 同步/异步、OpenAI Responses 同步/异步此前都是「判 `None` 后直接 `int(value)`」，那只挡住了缺字段。SDK 或兼容网关返回 `"abc"`／`1.5`／`[1]`／位数超限的数字串／`inf` 时，抛的是裸 `ValueError`／`TypeError`／`OverflowError`，与「token 计数」这件事无关，调用方也拿不到是哪个字段出的问题。现抽出 `require_token_count`：非负整数是这份数据的真实契约（token 计数不可能为负，也不可能带小数），八处统一改走它，形态异常一律报成带 provider 与字段名的 `RuntimeError`。

- 修复 provider 模块漏收无人守卫：十三个 provider 模块**没有任何静态导入路径**——它们只经 `importlib.import_module(provider_info["module"])` 按运行时字符串加载（`src/providers/factory.py`）。PyInstaller 的静态分析看不见这一层，因此构建脚本的 `HIDDEN_IMPORTS` 是它们在冻结包里能否被找到的**唯一**依据。两个清单分处两个文件、各自演化，而此前**没有任何守卫把两者绑在一起**，也没有任何构建期检查会碰这些模块——自检只走分片器与日志路径，不导入也不实例化任何 provider。后果与当初 `tiktoken_ext.openai_public` 漏收完全一致：PyInstaller 不报错、`--validate` 通过，用户在那个渠道上才拿到 `ModuleNotFoundError`；只是那次被发现了，这次没有守卫。现补两道防线：`run_smoke_test` 逐个导入注册表里的 provider 模块（只导入不实例化，因此不需要凭证），让漏收在**构建期**就被拦下；另加双向一致性用例，注册表新增模块而未同步清单、或清单残留已删除的模块名都会被判失败。

- 修复 PEM 私钥正文逃出脱敏边界：这是本项目**自己点名要防**的一类凭证——`SENSITIVE_OPTION_KEYS` 里同时列了 `privatekey`／`sshkey`／`sshprivatekey`，Google Vertex AI 的 ADC 服务账号 JSON 里就带 `private_key`。但键名规则只挡得住「键名 + 值」的形态：私钥**正文**作为一段独立文本出现时没有任何锚点。实测两条真实泄漏路径：`validate_secret_free_options` 把私钥当键名拒绝时，回显的消息里带着完整正文（该消息经 `safe_exception_text` 直接展示给用户）；裸 PEM 文本走 `redact_sensitive_text` 原样返回。现补一条 PEM 块规则，覆盖 RSA／通用／OpenSSH／EC 私钥，并同时匹配「BEGIN/END 成对」与「只有 BEGIN」两种形态——后者是必需的，`safe_exception_text` 会把消息截到 240 字符，END 标记可能已被切掉，只匹配成对形态会漏。证书与公钥不受影响（只匹配 `PRIVATE KEY`）。实测该规则对普通文本不增加可测开销（6k/60k 字符的对照基准由既有规则主导）。

- 修复旧日志清理漏掉轮转备份：写入侧是 `RotatingFileHandler(maxBytes=1MB, backupCount=5)`，单个文件超过 1 MB 就轮转出 `.log.1` ~ `.log.5`。清理用的文件名模式以 `$` 结尾，那些备份**一个都不匹配**，于是「清理旧日志」只删掉主文件，把绝大部分体积留在盘上——实测造一批 30 天前的日志（主文件 + 5 个备份），清理后仍剩 83% 的字节；本机真实日志目录里也躺着 2.0 MB 的备份、占该目录体积的 31%。`log_path` 是用户可配置项，指向磁盘时这份残留会一直累积，而 `cleanup_old_logs` 每次启动都跑。现把模式放宽为 `\.log(?:\.\d+)?$`，后缀限定为 `.` + 数字——`RotatingFileHandler.doRollover` 的备份名就是 `baseFilename + "." + 序号`，不会产生其它形态，因此 `app_log_2020-01-01.log.bak` 这类仍不受影响。

- **修复发布工作流的凭证检查被 SIGPIPE 静默跳过（本轮最严重）**：步骤用 `printf '%s\n' "${listing}" | grep -qE '/(\.env|config\.toml)$'`，而 `grep -q` 命中即退出、上游 `printf` 收到 SIGPIPE 以 141 退出，`set -o pipefail` 让整条管道返回 141——`if` 只把 0 当成功，于是**「命中」被判成「未命中」**。触发条件是归档列表超过管道缓冲区（Linux 上 64 KiB）：实测真实 v1.4.0 产物的列表已是 75–79 KB，也就是这条检查自发布起就一直失效。两个方向都与事实相反——含 `.env` 的包被放行发布（凭证泄漏），而 `tiktoken_cache` 检查会对正常包误报缺失。改用 `grep ... >/dev/null` 的 here-string（grep 读完整个输入，不产生 SIGPIPE）。既有 8 条工作流用例的合成归档只有 1–2 个条目、列表远小于缓冲区，因此完全测不到这条路径；新增两条 6000 条目的用例（一个含 `.env`、一个干净）把两个方向都钉住。
- **修复 `get_relative_path` 对所有取值套 `safe_repr` 导致相对路径功能失效**：上一轮为「兜底分支自己会抛」引入的修复把 `safe_repr` 用在了**参与计算**的值上，而 `safe_repr` 是给错误消息用的——它加一对引号并在 80 字符处截断。带引号的前缀永远匹配不到 `ROOT_DIR`，`os.path.relpath` 于是原样返回绝对路径（外加引号），超长路径还会丢掉文件名；真实知识库 39 个文档 100% 命中，三个调用点（召回测试、聊天、建库脚本）一起受影响。现改为 `_path_text`：只在 `str()` 真的失败时才降级到 `safe_repr`，正常路径走原样的 `str()`。同时把回归断言从「返回非空字符串」（原样返回的实现也能通过）换成「结果是相对形态且不含 `ROOT_DIR`」，并补一条超长路径不被截断、不加引号的用例。
- **修复 Excel 日志写入成功却打不开**：xlsx 是 XML，**两处**各自拒绝不同的码位——`openpyxl` 的 `check_string()` 在 `append()` 期拒绝 C0 控制符（已处理），而 XML 1.0 不允许的码位在 `save()` 期被 `xml.etree` 写成非法字符引用：**写入成功、读回失败**，`load_workbook` 抛 `ParseError`，而 `log_results` 的 `except Exception` 把它吞成一行日志——用户看到「召回测试完成」而文件打不开，而这份日志是召回测试的**唯一产出**。第 2 类漏掉的正是孤立代理项（`U+D800`–`U+DFFF`）与 `U+FFFE`／`U+FFFF`：前者经 legacy pickle 路径可达（pickle 是二进制格式、不受 UTF-8 约束），后者是**合法 UTF-8 码位**（`read_text(encoding="utf-8")` 与 `BasicCleaner` 的 `\s+` 折叠都放行）。现把两类一并补进替换表，并断言 `load_workbook` 能读回（只断言 `save()` 不抛是假绿——原实现正是「写得进、读不出」），另加一条反向守卫确认合法码位（emoji／CJK／`U+D7FF`／`U+E000` 边界）逐字保留。
- **修复嵌入 provider key 打错字抛裸 `KeyError`**：`default_embedding_provider` 没有任何 validator（`Settings` 里只是普通 `str`），而 `EmbeddingService.embedding_model_detail` 与 `embed_knowledge_base.py` 的展示都直接下标。实测 `DEFAULT_EMBEDDING_PROVIDER=gogle` 时脚本以裸 traceback 退出、消息里连可选值都没有，而同一份配置走 `ModelProviderFactory` 得到的是带可选值列表的 `ValueError`——两条路径的报错契约本该一致，否则调用方按 `ValueError` 写的错误分类与脱敏接不住 `KeyError`。调用点在检索与建库的主路径上（`embed_query` 的 Google 分支、`knowledge_build_service` 写清单）。现两处共用 `_require_model_detail`。
- **修复 `_resolve_version` 漏接 `TypeError`**：`except (OSError, KeyError, tomllib.TOMLDecodeError)` 少了 `TypeError`，而 `["project"]`／`["version"]` 在形态不对时抛的正是它（`project = "x"` → `string indices must be integers`；`project = 5` → `'int' object is not subscriptable`），不是 `KeyError`。本函数在**模块导入期**执行，漏出的异常让进程以裸 traceback 退出——而两级兜底存在的意义正是「读不到就降级」，与已修的 `VERSION.txt` 解码失败同一失效模式。现补上 `TypeError` 与 `ValueError`，并按四种畸形 `pyproject.toml` 形态各加一条子进程用例。
- **修复基础清洗器哨兵长度的平方复杂度**：`while placeholder in content: placeholder += "\x00"` 每次 `in` 都是一次全串扫描、每次 `+=` 又让哨兵长一位，于是 N 个 NUL 要扫 N 次——实测 20 万个 NUL 耗时 19.8 秒、100 万按平方外推约 1300 秒，而旧的字面量哨兵是 O(1)。触发输入很现实：零填充的 md（写入中断／预分配／误拷的二进制残片）或 Windows 用户存成 UTF-16LE 的纯 ASCII 文档，`read_text(encoding="utf-8")` 对两者**都不报错**，因此 `Pipeline._process_file` 的解码失败分支拦不住，用户只看到「向量化处理」卡死数分钟且无任何提示。现改为「一次扫描取一个内容里不存在的非空白字符」。
- **修复清洗器哨兵在边界上的歧义（同一处修复自身引入的缺陷）**：把哨兵写成「最长 NUL 游程 + 1」这种多字符形态看似更省事，实则在边界上歧义——内容 `"a\x00\x00\x00\n\nb"` 换出 4 个 NUL 后与原有 3 个 NUL 连成 7 个，`str.replace` 取最左匹配、只吃掉前 4 个，剩下 3 个 NUL 被留在双换行**之后**，实测得到 `'a\n\n\x00\x00\x00b'`：用户内容被静默改写。单个且内容中不存在的字符不可能有这种歧义。已在 2 万条随机输入上与修复前的实现做差分（含 NUL 游程、Unicode 空白、控制字符），结果零差异。
- **修复 `RecursiveTextSplitter` 的 `mode` 静默回退与尺寸参数无类型守卫**：`mode` 原先只判 `== "token"`，任何拼错／大小写不符／非字符串的取值都**静默退回 char 模式**——实测 `"Token"`／`"tokens"`／`None`／`["token"]`／`123` 全部得到 char 分片，于是 `chunk_size=1500` 意味着 1500 **字符**而不是 1500 token（中文下每 token 约一字），分片规模与配置意图不符且无任何提示，违反 AGENTS.md「不用静默回退或占位结果掩盖失败」。尺寸参数则只做比较、不做类型检查：`child_chunk_size='300'` 构造成功、到 `split()` 才抛 `TypeError: '>=' not supported between instances of 'int' and 'str'`，而本模块的契约是 `ValueError`。现两者都在构造期校验，另修掉 `parent_chunk_size=0` 时 `or` 兜底与 `is not None` 比较造成的「校验的值和用的值不是同一个」（实测报出 `parent=0` 而真正生效的是 1500）。
- **修复分隔符表无长度上界导致 `RecursionError`**：LangChain 的 `_split_text` 按分隔符表**递归**、每层消耗一项，实测 1500 项抛 `RecursionError: maximum recursion depth exceeded`——它不是 `ValueError`，穿透调用方的错误分类与脱敏。表来自用户可写的 `kb_splitter_separators`（TOML 数组），`Settings` 的 validator 只做 `str -> list` 拆分、没有长度上界。现加 64 项上界（真实配置是个位数，默认只有 `["###"]`）。
- **修复 `SessionConfig` 的字段名校验漏出 `TypeError` 与位数上限错误**：`key not in self.to_dict()` 对不可哈希的 key 抛 `TypeError: unhashable type`，而消息里的 `{key!r}` 对位数超限的整数抛 `ValueError: Exceeds the limit`——两条都与「字段名写错了」无关，而同文件的 `_validate_fusion_strategy` 早已认领同一类线索，只有字段名这一处漏了。`set_many` 同样（且只修 `__setitem__` 会留下另一个入口）。现两处共用 `_require_known_field`，未知的**字符串**字段仍照常报 `KeyError`。
- **修复组合校验读取属性直写值时抛 `OverflowError`**：`SessionConfig` 是公开类，普通属性没有 `__setattr__` 守卫，嵌入方（或携带超大整数的 pickle 快照）可以直接 `cfg.vector_weight = 10**400`。随后任一次 `set_many`／`__setitem__` 都会读到该兄弟字段并运算：`float(10**400)` 抛 `OverflowError`（`ArithmeticError` 子类，**不是** `ValueError`）、`int(inf)` 同样、`int(nan)` 抛 `ValueError`——前两条会穿透只接 `ValueError` 的调用方。同文件注释曾断言「到这一步必是有限浮点、不会再抛 `OverflowError`」，那个前提对属性直写不成立。现经 `_require_finite_number`／`_require_int_value` 收敛，校验与运算用同一个值。
- **修复候选规模过大的报错消息自己会抛**：`_reject_oversized_candidate_pool` 的消息插值 `{candidates}`（= `top_k * multiplier`），而 `top_k` 被直写成位数超限整数时，它在**正要报「候选规模过大」**的那一步抛 `ValueError: Exceeds the limit`——类型对但内容与字段毫无关系。同文件的 `_require_bounded_int` 早已用 `safe_repr`，只有这里漏了。
- **修复快照清单的写入侧仍用 `str()`／`int()`**：`from_mapping`（读取侧）早已按「pickle 快照可携带位数超限整数」修过，但 `to_toml`（写入侧）写的是裸 `str(value)` 与 `int(value)`——同一条缺陷只修了一个方向。实测文本字段是 `10**4301` 时 `write_manifest` 抛「整数转字符串超限」，与「快照写入失败」毫无关系；计数字段是 `inf` 时 `int()` 抛 `OverflowError`（`ArithmeticError` 子类，不是 `ValueError`）。现两处都经与读取侧同源的收敛，并给计数加 64 位上界（TOML 能表示任意精度整数，但这个量级的计数没有任何现实意义）。
- **修复 `/config` 应用失败终止整个聊天会话**：内层 `try` 只接 `(KeyboardInterrupt, EOFError)`，而 `SessionConfig` 的校验器会拒绝非法取值（`ValueError`）。配置编辑器的动态上界与 `SessionConfig` 的守卫是两套独立实现，任一侧演化就会命中这条路径——实测编辑器返回「非法字段 + 非法值」时异常穿透到 `finally`，用户被丢回主菜单并**丢失会话**。同文件的 `chat_async` 对异常是恢复的（打印后继续下一轮），这里与之对齐：报错（经 `safe_exception_text` 脱敏）、保留原配置、回到提示符。诊断日志经 `getattr` 取，避免日志取不到时把恢复路径变成新的失败点。
- **修复召回测试日志在同一秒内互相覆盖**：文件名的时间戳只到**秒**（`recall_test_log_%Y%m%d_%H%M%S.xlsx`），同一秒内构造的第二个 `ExcelLogger` 会拿到同一个路径并覆盖前一个文件——实测两个实例的 `filepath` 完全相同。而 Excel 日志是召回测试的**唯一产出**，覆盖等于悄悄丢掉一次测试的结果。可达路径是 `main.py` 的菜单循环（从召回测试返回主菜单后再次进入）或嵌入方在同一秒内构造两个 logger。现用 `O_CREAT | O_EXCL` 独占创建，冲突时加 `_1`／`_2` 后缀（第一个仍是原文件名，保持既有形态不变）。判据用 `O_EXCL` 而不是 `os.path.exists`：后者是「先查后写」，两个进程同时构造时会双双认为路径可用、然后互相覆盖。构造中途失败时回收已创建的占位文件，否则目录里会留下一个 0 字节的 `.xlsx`，看起来像一次「没有结果的召回测试」。
- **修复分片器的日志口径**：标准分支记的是 `len(langchain_chunks)`（**过滤前**的数量），一份被整段判空的文档会显示「生成 8 个块」而实际贡献 0 个——日志反过来掩盖了问题。实测 `"a" + "\t"*5 + "b"` 在 `chunk_size=1` 下过滤前 7 个、实际收录 2 个。现改为记录**实际收录**的块数，空文档单独记成「清洗后为空，跳过」而不是「生成 0 个块」。「原文非空却产出 0 块」经笛卡尔扫描（16 种输入 × 5 组分隔符 × 3 组尺寸 × 2 种结构）未复现，故**不**为它写告警分支——写不到的死代码只会让人误以为有防护。

- **修复 `redact_sensitive_text` 的非字符串分支自己会抛**：该函数第一个分支是 `if not isinstance(value, str): value = str(value)`——明确声明支持非字符串，而 `str(10**4301)` 撞上 CPython 的整数位数上限抛 `ValueError`。同文件的 `safe_exception_text` 早就为同一问题用 `safe_repr` 兜底（其 docstring 还专门解释了 pickle 为什么能带进这种值），只有这里漏了。现内联同一判据（位数超限时用 `bit_length()` 给出量级）；**不**导入 `config.safe_repr`，因为 `config` 反过来从本模块导入 `redact_sensitive_text`，跨模块引用会形成循环导入。

- **修复空的路径配置静默解析成「当前工作目录」**：`resolve_path` 的 docstring 承诺「将相对路径解析为绝对路径」，但解析**之前**有一个 `if not v: return v` 提前返回，而 `Path("")` 在求值期等价于 `Path(".")`。两个后果：其一，**pickle 的信任根被扩到 CWD**——`FaissStore.load_snapshot` 拿 `snapshot_root` 当来源信任根（`AGENTS.md` 要求快照只从本项目生成的文件加载），实测 `SNAPSHOT_ROOT=""` 时信任根落在仓库根目录，在其中放一个带 `ACTIVE_SNAPSHOT` 标记的目录即可让 `load_snapshot` 接受并反序列化它的 `chunks.pkl`，边界等于不存在；其二，**行为随 CWD 漂移**——五个路径字段都原样返回，同一份配置在仓库内与仓库外运行指向不同目录（实测 CWD=/tmp/xxx 时五者全部解析到 /tmp/xxx），而默认值（如 `data/kb`）走 `ROOT_DIR` 分支、不随 CWD 变。现显式拒绝空值与纯空白（`Path("   ")` 会解析成一个名为空格的**真实目录**），并给出「删除该配置项回落默认值」的可执行指引。`cache_path` 清理侧的 `if not cache_dir` 守卫是删除前的最后一道防线，不是「空值合法」的依据。经 `.env` 与 `config.toml` 两条来源实测可达。
- **修复清洗器复杂度用例的间歇误报**：该用例用 2k 个 NUL 做基准、断言 100 倍规模的耗时比低于 600，而 2k 的耗时约 0.1ms，已接近 `time.perf_counter()` 的可靠量级——实测同一台机器上 8 次采样出现 967 与 598 两个越界值，稳态比值则在 50~75 之间波动。现基准改用 20k（约 0.7ms）、规模放大改为 10 倍、每档取多次采样的最小值（逼近无干扰耗时，而非把 GC 与调度抖动算进去），阈值取 60：线性实现实测比值 7.8~11.5，平方实现实测 700，两侧各有 5 倍以上余量。定向变异（把哨兵改回 `while placeholder in content`）后该用例无法通过。

- **修复空的 Base URL 静默变成相对路径**：`openai_compatible.__init__` 用 `getattr(settings, "<provider>_base_url")` 取值，随后只判 `if provider != "openai" and not self._base_url`——**openai 被显式排除在外**（它允许回落 SDK 默认地址）。但字段默认值非空、`.env` 里又有 `OPENAI_API_BASE`，于是「显式置空」这条路径无人把守：实测 `OPENAI_API_BASE=""` 时 `self._base_url` 是空串而非 `None`，判 `is None` 的回落分支不触发，`openai.OpenAI(base_url="")` 得到 `URL('')`，请求 URL 变成相对路径 `chat/completions`（没有主机名），失败信息与「地址配错」毫无关系；`"   "` 更会被 SDK 编码成 `%20%20%20/`。现拒绝字符串形态的空值，同时保留 `None` 的原有语义（未配置 → 用 SDK 默认地址，实测为 `https://api.openai.com/v1/`），并为两条方向各补一条断言。错误消息按取值**实际来自哪个字段**命名：openai 渠道的现行字段是旧名 `openai_api_base`，只报 `openai_base_url` 会把用户指向一个不存在的配置项。

- **修复 rerank 缓存键在集合与自引用容器上失效**：`_stable_fingerprint` 的 docstring 承诺「递归到叶子」与「绝不抛异常」，实测两条都只在 dict/list/tuple 上成立。其一，集合走 `repr()` 兜底、**没有递归**，而集合含位数超限的整数时 `repr()` 抛 `ValueError`，整个集合塌成 `<set>`——实测 `ModelDetail(options={"x": {10**4301}})` 与 `{"x": {10**4302}}` 都通过 `validate_secret_free_options`（它只查敏感键名、不做递归类型校验），随后两者得到同一个缓存键，两份不同的 rerank 配置静默复用同一个 provider 实例：错误配置的请求打到错误的端点上，且没有任何报错，比崩溃更糟。其二，自引用容器让它无限递归抛 `RecursionError`，而 `RecursionError` 继承 `RuntimeError`、不在调用方 `except (TypeError, ValueError)` 的范围内，会穿透成未脱敏的 traceback。现集合分支一并递归（元素按各自渲染结果排序——集合无序且 `repr` 顺序受哈希随机化影响，不排序会让**同一个**取值在两次调用间产生不同指纹、缓存永远命中不了），并沿当前路径记录容器 `id` 以截断环（用「路径上的 id 集合」而非全局去重：同一对象在兄弟位置出现两次是正常取值，只有回到祖先才是环）。经 50 次重复调用验证同值同指纹，且跨进程哈希种子一致。

- **修复检索方法的 `top_k` 无校验且两处判据不一致**：`semantic_search` 与 `keyword_search` 同名参数、同一份上界来源，此前判据完全不同，同一取值给出三种收场。其一，`top_k=0`／负数：前者触发 FAISS 内建 `assert k > 0`，抛的是**没有消息的** `builtins.AssertionError`（实测 `str(e)` 为空串，栈里只有 C 层，用户既不知道哪个参数错了也不知道怎么改；`python -O` 下断言被剥掉、行为还会变成未定义）；后者走 `if len(results) >= top_k: break`，而循环体先 `append` 再判长度，于是 **`top_k=0` 返回 1 条**——调用方要求「不要结果」却拿到一条且无任何提示。其二，非整数：FAISS 报 `TypeError: an integer is required`、后者在 `int >= str` 处报 `TypeError: '>=' not supported`，两者都是裸 `TypeError`，按 `except ValueError` 写的调用方接不住。其三，过大：FAISS 不报错也不截断，直接按 k 分配结果数组——实测 `top_k=10**9` 挂起（>30s）、`10**8` 耗时 15.3s。配置层的 `_MAX_RETRIEVAL_TOP_K` 拦不住这三个方向：它只作用在 `Settings.chat_top_k` 与 `SessionConfig.top_k` 上，而这两个方法是 `VectorStoreBase` 的公开方法（库调用方与测试直接调用），参数一路透传到 FAISS。现抽出 `_require_top_k` 作为唯一判据、两处共用，`bool` 显式排除（它是 `int` 子类，`True` 会被当成 1 放行），并补一条「`top_k` 大于文档数仍合法」的反向守卫防止上界写成「不得超过命中数」。回归用例的夹具先自检「确实存在正分文档」——语料全为 0 分时 `keyword_search` 会在最高分处提前返回空列表，把 `top_k` 的问题整个遮住。

- **修复父块内容把非字符串原样写进文档正文**：`_promote_parent_context` 把 `resolver(parent_id)` 直接赋给 `promoted_doc["page_content"]`，只判了真值（`if parent_content:`）、没判类型，而 `page_content: str` 是 `Document` 的契约（`src/models/document.py`），`FaissStore._validate_loaded_document` 在快照载入期也按这条校验。可达性：`parent_resolver` 取自 `getattr(self.vector_store, "resolve_parent_content", None)`，而 `VectorStoreBase` 正是为替换实现而存在的——签名声明 `-> str | None`，第三方实现返回别的形态不会被任何地方拦下。实测返回列表让 `page_content` 变成 `['a', 'b']`、返回整数变成 `123`：展示层 `document_content` 把它收敛成 `"N/A"`（用户看到「无内容」而不是报错），而 prompt 构造对正文做 `.lower()` 时抛 `AttributeError`，检索整体不可用，报错现场离真正的原因很远。现经 `_key_text` 收敛；**不**用裸 `str()`——位数超限的整数（pickle 是二进制格式、不受十进制位数限制，畸形快照能带进来）会让它抛 `ValueError: Exceeds the limit`，而本函数的职责是「把父块内容放进正文」，不该在正要成功时崩。

- **修复协议名归一化在形态异常的取值上抛裸的整数转换错误**：五个归一化点都直接写 `str(value).strip().lower().replace("-", "_")`，而 `10**4301` 撞上 CPython 的整数位数上限，抛的是 `ValueError: Exceeds the limit (4300 digits) for integer string conversion`——既没有「哪个参数错了」，也没有「可选值是什么」，用户拿到的是与协议诊断毫无关系的整数转换错误。其中 `protocol_status`／`capability_report` 是公开的诊断入口（配置界面与排障工具会调），参数来自调用方而非 `Settings`——`ModelDetail.protocol` 有 validator，但这两个入口收的是**裸取值**，没有任何校验；`openai_compatible._normalize_protocol` 与 `VolcengineProvider.__init__` 经公开构造器可达。现统一显式拒绝非字符串并给出参数名与可选值；`_responses_provider_variant` 例外——它只做 Ark/OpenAI 的**分流**，非字符串一律返回 `"openai"` 而不抛错，因为它抛在 `_normalize_responses_native_item` 的入口，会把本该给出的「字段类型不对」消息换成一条与协议无关的整数转换错误。

- **修复 Anthropic 流式拒答文本在超大整数上崩溃**：`_stream_event` 的两处拒答分支写的是 `refusal if isinstance(refusal, str) else str(refusal or "")`，非字符串走裸 `str()`。而 `refusal` 来自服务端返回的 JSON（`content_block` 或 `delta` 里的字段），形态不受本地控制——JSON 里一个超大整数字面量（`10**4301`）会让 `str()` 撞上 CPython 的整数位数上限，抛 `ValueError: Exceeds the limit`。该异常发生在**流式解析**过程中（`_iterate_stream_events` 的生成器里）：用户已经看到部分输出，随后拿到一条与「服务端拒答」毫无关系的整数转换错误；而同一函数对 `42` 正常转成 `"42"`、对 `None` 给出 `""`，只有位数超限的那一个炸——收场方式取决于取值的**大小**而不是形态。现抽出 `_refusal_text`，非字符串给 `safe_repr` 的可读描述而不是空串（空串会让调用方误以为「没有拒答」），两处共用同一判据。

- **修复 Anthropic 拒答详情（`stop_details`）在超大整数上崩溃，且错误掩盖安全拦截信号**：`_extract_result` 与 `_stream_event` 的 `message_delta` 分支各有一处`str(explanation or category or "<默认文案>")`，非字符串走裸 `str()`。`stop_details` 是 Anthropic 的拒答详情对象，`explanation`／`category` 都来自服务端 JSON、形态不受本地控制——JSON 里一个超大整数字面量（`10**4301`）就让 `str()` 撞上 CPython 的整数位数上限，抛 `ValueError: Exceeds the limit`。后果比单纯的崩溃更糟：`stop_reason == "refusal"` 是**安全拦截**的信号，此处抛异常会把一次本该被显式上报的拒答变成一条与安全策略无关的整数转换错误——用户看到的是「程序出错了」，而不是「请求被安全策略拒绝」。现新增 `_refusal_detail`（与上一轮的 `_refusal_text` 同一判据，非字符串给 `safe_repr` 描述），两处共用。空值语义有意与 `_refusal_text` 不同：`explanation`／`category` 可以合法缺失，`None` 必须返回空串以继续向后回落，否则缺失的 `explanation` 会渲染成 `"None"` 并挡住 `category`；「两者皆缺仍给默认文案」这条由上层保留。

- **修复 `finish_reason` 归一化在超大整数上抛裸的整数转换错误**：`GoogleProvider._normalize_finish_reason` 的 `str(raw)` 与 `AnthropicProvider._stream_event` 的 `str(reason)` 都直接转换服务端返回的停止原因。该字段来自响应 JSON、形态不受本地控制，`10**4301` 撞上 CPython 的整数位数上限抛 `ValueError: Exceeds the limit`——而两处都在**响应解析**里，用户已经拿到部分输出后才看到一条与「停止原因」毫无关系的整数转换错误。现非字符串走 `safe_repr`（给出量级描述，仍可区分不同取值）。修 Google 时发现一处交互缺陷：紧随其后的 `if "." in text: text.rsplit(".", 1)[-1]` 是给 `"FinishReason.MAX_TOKENS"` 这类带模块前缀的枚举名准备的，而 `safe_repr` 的截断标记恰好也是 `...`——对它套这条逻辑会把整段数字切光、只剩空串。现该逻辑只对字符串形态生效。

- **修复 `extra_headers` 的取值校验与 embedding 配置取值的校验（第三轮对抗审查，逐条实测复现）**：

  - **Anthropic 的 `extra_headers` 只校验容器、不校验取值**：`validate_secret_free_request_overrides` 确认整体是 Mapping 后原样保留，实测 `CompletionRequest(extra_headers={"anthropic-beta": 42})` 构造成功。非字符串值的两种收场都不在渠道的错误契约内——交给 httpx 时抛 `TypeError: Header value must be str or bytes, not <class 'int'>`（实测 `httpx.Headers({"x-trace": None})` 同一条），而本模块对 `anthropic-beta` 做 `.split(",")` 时抛 `AttributeError: 'int' object has no attribute 'split'`（`list`／`dict` 同理，`bytes` 则抛 `TypeError: a bytes-like object is required`）。现新增 `_require_string_header_values`：非字符串给带头名的 `ValueError`，`None` **剔除**（剔除后与「该头未设置」完全等价，符合调用方写 `extra_headers={"x": None}` 时的意图）。校验放在 `_user_profile_fields` 的两条分支**之前**——`user=None` 时该函数原本直接返回原样的 `extra_headers`，而这是最常见的调用形态，比下面的合并分支更容易走到。
  - **`anthropic-beta` 头值写 `None` 时抛裸 `AttributeError`**：合并分支对它做 `.split(",")`，实测 `_user_profile_fields("u1", {"anthropic-beta": None}, None, beta=False)` 抛 `AttributeError: 'NoneType' object has no attribute 'split'`。现于合并前剔除 `None` 值，语义与「该头未设置」等价。合并同时改为写在副本上——原实现就地修改调用方的 `request.extra_headers`。
  - **beta 计数入口丢弃了校验后的头**：`beta_count_tokens`／`async_beta_count_tokens` 把 `_user_profile_fields` 的返回值丢进 `_`，转头写回原始的 `request.extra_headers`——被剔除的 `None` 值于是原样发给 httpx。同文件 `beta=False` 的计数入口用的正是 `profile_headers`，两处现对齐。
  - **`EmbeddingService.embedding_model_detail` 只判键在不在、不判值可用**：`{"local-hash": None}` 的键确实存在，`None` 被原样返回，两个调用点（`embed_query` 的 Google 分支、`knowledge_build_service` 写清单）紧接着读 `.provider`／`.model_name`，在那里抛裸 `AttributeError`。同一份配置走 `ModelProviderFactory.get_embedding_provider` 得到的是带字段名的 `ValueError`——两条路径的报错契约本该一致，否则调用方按 `ValueError` 写的错误分类与日志脱敏接不住 `AttributeError`。现复用 `require_detail_attributes`（判据是属性存在性而非 `isinstance(..., ModelDetail)`，理由见其 docstring——使用方只要求取值提供这几个属性，仓库测试就用 `SimpleNamespace` 构造等价配置）。可达路径里最晚暴露的是**零分块**的 markdown：它跳过 embedding 调用、直接走到写清单那一步读 `model_name`。

- **补齐 Anthropic `user`／`betas` 的取值校验（第四轮对抗审查）**。`CompletionRequest` 是普通 dataclass，`__post_init__` 只校验 `extra_headers`／`extra_query`／`extra_body`，声明为 `str | None` 与 `Sequence[str] | None` 的这两个字段不受任何约束。它们都进 HTTP 请求（`user` 转成 `user_profile_id` 头、`betas` 拼成 `anthropic-beta` 头），走真实 SDK（`httpx2.MockTransport` 拦网）实测修复前的收场：

  - `invoke(user=5)`／`beta_create(user=5)` → `AttributeError: 'int' object has no attribute 'encode'`（`['a']` 与 `b"x"` 同理，后者是 httpx 对 `bytes` 头值走另一条分支）。`AttributeError` 不落本渠道的 `except ValueError`／`except RuntimeError` 分类。七个入口（`invoke`／`count_tokens`／`beta_create`／`beta_stream`／`beta_count_tokens` 及三个异步对应项）实测全部如此。
  - `beta_create(betas=5)` → 裸 `TypeError: 'int' object is not iterable`。
  - `beta_create(betas=[1, 2])` 发出 `anthropic-beta: 1,2`；`betas=b"x"` 发出 `120`（`list(b"x")` 得 `[120]`）；`betas={"a": 1}` 静默取**键**发出 `a`；`betas="s"` 被 `list()` 拆成 `['s']`。服务端收到不存在的 beta 名——要么报一个与调用方写法毫无关系的错，要么静默忽略掉本想启用的特性。
  - 现新增 `_require_profile_id` 与 `_require_beta_tokens`：前者只接受 `str | None`；后者把 `str`／`bytes`／`Mapping` 三类「可迭代但不是字符串列表」的形态显式拒绝（`list()` 对它们不报错，只会静默给出错误结果），并逐个校验元素类型。
  - **`betas` 的校验必须放在 `_build_message_params` 的 `beta_values` 构造处**，不能只放在 `_user_profile_fields` 里：那个函数仅在 `user is not None` 时被调用，而 `betas` 走 `beta_values` 直接进 params——只加在前者时 `beta_create(prompt="hi", betas=5)` 仍抛裸 `TypeError`（这正是首版实现漏掉的路径，回归用例 `test_anthropic_betas_validated_without_user` 专门盯它）。

- **补齐工具定义、消息列表、rerank 与 embedding 入口的入参形态校验（第五轮对抗审查）**。四组缺陷同一类：判据只覆盖了「可迭代但元素非法」，漏了「不可迭代」那一半，或者把形态判据放在了读取形态的代码**之后**。

  - **四个 provider 的 `_convert_tools` 对不可迭代的 `tools` 抛裸 `TypeError`**：`google`／`anthropic`／`volcengine`／`openai_compatible` 都写 `for index, tool in enumerate(tools)`，而同函数对 `"s"`／`{"a": 1}`／`b"x"` 这类可迭代但元素非法的形态早已给带序号的 `ValueError`——只有标量漏了。实测 `google.invoke(tools=5)` 与 `openai.complete(tools=5)`（responses 协议）抛 `TypeError: 'int' object is not iterable`。现四处统一补容器判据。
  - **`extract_system_prompt` 的形态判据缺失**：它是 12 处调用的共享 helper，且在多数 provider 里**先于** `normalize_messages`（后者有完整判据）被调用，因此非法取值在这里就炸。实测 `messages=5` 抛裸 `TypeError: 'int' object is not iterable`，`messages="s"` 抛裸 `AttributeError: 'str' object has no attribute 'get'`——字符串是合法的 `Sequence`，只判可迭代拦不住。现判据与 `normalize_messages` 逐字对齐（含 `bytes`／`bytearray` 的显式排除），条目非 Mapping 时给带下标的 `ValueError` 而不是静默跳过——跳过会让一条写错的消息**悄悄消失**，system 提示随之丢失。
  - **两个 rerank provider 把 `len(documents)` 写在参数校验之前**：`jina` 与 `siliconflow_rerank` 的同步/异步四个入口，日志行 `文档数: {len(documents)}` 都在 `_validate_inputs` 之前求值。实测 `documents=5` 抛裸 `TypeError: object of type 'int' has no len()`，而紧随其后的校验对同一取值给的是带指引的 `ValueError`。现把校验提到日志之前（`_prepare_payload` 里那次保留——它是请求构造路径的守卫）。
  - **六个 embedding 实现都不判 `texts` 的形态**：`TextEmbeddingModel` 的全部公开入口此前没有一处校验，而 `texts` 是它们唯一的必填入参。`texts=5` 在四处抛裸 `TypeError: object of type 'int' has no len()`、在 `local_hash` 抛 `TypeError: 'int' object is not iterable`；`texts="abc"` 更隐蔽——字符串是合法的 `Sequence`，被**静默**当成三条单字符文本向量化（`local_hash` 实测返回 3 条），调用方以为传了一条文本、实际入库三条垃圾向量。现新增 `require_text_list`（`src/providers/__base__/model_provider.py`），接入 `local_hash`／`google`／`openai_compatible`／`volcengine` 的同步与异步共 8 个入口。空列表**合法**——语义是「没有文档要向量化」，与 rerank 的「必须非空」有意不同。


- **补齐 ETL 三个阶段的入参形态校验（第六轮对抗审查）**。``Pipeline.process`` → ``Cleaner.clean`` → ``Splitter.split`` 是建库主路径上相邻的三步（``KnowledgeBuildService._process_file`` 对知识库里的每个文件各调一次），三处此前都不判入参形态，实测：

  - ``Pipeline.process(5)`` → 裸 ``AttributeError: 'int' object has no attribute 'metadata'``（方法第一行就读 ``document.metadata``）。``"abc"``／``[5]``／``None`` 同理。``process`` 是 ``_process_file`` 的直接下游，异常落在 ``KnowledgeBuildService.build`` 按 ``ValueError`` 建立的分类之外。
  - ``BasicCleaner.clean(5)`` → 裸 ``TypeError: 'int' object is not iterable``；``clean("abc")``／``clean([5])``／``clean([None])`` → 裸 ``AttributeError: 'str' object has no attribute 'content'``。
  - ``RecursiveTextSplitter.split(5)`` → 裸 ``TypeError: object of type 'int' has no len()``（日志行里的 ``len(documents)``）；``split("abc")``／``split([5])`` → 裸 ``AttributeError: 'str' object has no attribute 'metadata'``。``split_hierarchical`` 是公开方法且同样暴露：``split_hierarchical(5)`` → ``TypeError: 'int' object is not iterable``。

  三处的 ``"abc"`` 尤其要紧：字符串是合法的 ``Sequence``，只判「可迭代」拦不住，它会被当成一串字符逐字处理。现三处共用同一判据（容器必须是 ``Sequence`` 且显式排除 ``str``／``bytes``／``bytearray``，条目必须是 ``Document`` 实例并带下标），分片器把判据抽成 ``_require_documents`` 供 ``split`` 与 ``split_hierarchical`` 共用——只守卫 ``split`` 会留下「直接调 ``split_hierarchical``」的口子。条目判据用 ``Document`` 实例而非鸭子类型：三个阶段后续都直接读它的 ``content``／``metadata``，而 ``Document`` 是 pydantic 模型（构造期已校验 ``content: str``），要求实例化才能进入能让错误定位到构造处而不是处理处。


- **补齐配置写入入口的容器形态校验**：`SessionConfig.set_many` 与 `Chatbot.chat_config` setter 的标注是映射，但运行时不受约束，非映射取值一路走到 `.items()`／`set_many` 的迭代才炸。实测 `cfg.set_many(5)` 与 `bot.chat_config = 5` 都得到裸 `TypeError: 'int' object is not iterable`，而 `cfg.set_many("abc")` 更糟——字符串是合法的可迭代对象，它被拆成一串单字符字段名，抛出 `KeyError: "SessionConfig 没有字段 'a'。"`（`bot.chat_config = "abc"` 同理）。同一个「配置写错」的收场不该取决于写错的是标量还是映射：非法**字段名**早已给带指引的 `ValueError`，现在非法**容器**也是。两处都在任何写入之前判形态（`str`／`bytes`／`bytearray` 显式排除），`chat_config` setter 的既有原子性契约（`deepcopy` 副本上写入）保持不变。

- **补齐 `retrieval_method` 的取值收敛**：该字段声明为 `RetrievalMethod`，但 `SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫，嵌入方（或携带异常取值的 pickle 快照）可以直写成任何形态。三个取用点此前各自假设它是枚举成员：

  - **`RetrievalService.retrieve` 静默走错检索路径**（最要紧的一条）：`retrieve` 用 `==` 判分支，而它是该函数里**唯一没有形态收敛**的字段——同一函数里 `top_k`、候选倍率、`score_threshold`、`rerank_enabled`、`hybrid_fusion_strategy` 都有对应的 `_coerce_*` 助手。实测 `cfg.retrieval_method = "hybrid"`／`None`／`5` 全部静默落进 `else` 分支，**退化成纯关键词检索**（记录调用可见只走了 `keyword_search`）；`"语义"` 同理。用户看到的是「混合检索配好了但只召回关键词命中」，而不是一条配置错误。`"hybrid"` 这条尤其容易踩：它是 `"混合检索"` 的英文写法。
  - **`display_chat_config` 在启动横幅处终止会话**：该函数读 `.value`，而它在 `start_chat_session_async` 里于任何提示符**之前**被调用。实测直写 `5`／`None` 抛裸 `AttributeError: 'int' object has no attribute 'value'`——用户刚看到「知识快照加载成功」，会话随即终止。
  - **`config_menu.edit_retrieval_params` 读同一个 `.value`**，菜单根本渲染不出来。

  现新增共享助手 `require_retrieval_method`（`src/utils/config.py`，`RetrievalMethod` 的归属地），三个取用点与 `SessionConfig._validate_retrieval_method` 共用它——配置边界与运行期读取侧从此只有一处判据、一种消息。判据按**属性存在性**而不是具体类型（与同文件 `require_detail_attributes` 同一约定：使用方只要求取值提供 `.value`，任何具备它的对象都能工作，仓库既有测试就用 `SimpleNamespace(value="混合检索")` 作替身）。接受枚举成员与它的字符串取值（`config.toml` 的 `chat_retrieval_method` 走的就是后者）；**不做**大小写／别名归一——静默接受 `"hybrid"` 等于把「配置写错了」变成「检索行为悄悄变了」。

- **补齐 `chat_temperature` 的取值收敛**：该字段有**完整**的校验器（`Settings.validate_chat_temperature` 与 `SessionConfig._validate_chat_temperature` 各一份，判据一致：拒 `bool`、非数值、非有限、负数、超出双精度的整数），但两个**读取点**都直读原属性，而 `SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫：

  - **`ChatService.generate_reply` 把非法取值静默送进请求**：`session_config.chat_temperature` 直接进 `CompletionRequest`，再原样成为 SDK 的 kwargs。实测 `"abc"`／`None`／`[0.5]`／`True`／`-1` 全部**静默通过**——而它们正是 `session_config["chat_temperature"] = ...` 会明确拒绝的取值。同一个字段「写得进就合法、直写就非法」两种口径；非法值要到服务端才暴露，用户看到的是接口返回的参数错误，而不是一条本地配置错误。
  - **`Chatbot.chat_async` 的兜底分支**（`chat_service` 缺失时）读 `self.session_config["chat_temperature"]`，同一个值、同一后果。

  现新增共享助手 `require_chat_temperature`（`src/utils/config.py`），两个读取点与 `SessionConfig._validate_chat_temperature` 共用它——与上一轮的 `require_retrieval_method` 同一模式。

- **修正 `_should_apply_score_threshold` 兜底分支的判据方向**：该方法的 docstring 写着「`fusion_strategy` 由调用方传入**已收敛**的取值：`retrieve` 已经校验过，这里再读一次原属性会绕过那道校验」，但它的**兜底分支**恰恰就在读原属性。判据是 `strategy == "rrf"`，而 `"RRF"`／`" rrf "`／`"Rrf"` 走的是 RRF 融合却不等于这个字面量——阈值过滤被错误地套用在排名倒数的量纲上（结果被静默清空），正是 `_coerce_fusion_strategy` 的 docstring 记录过的那个缺陷，只是留在了库调用方这条路径上。实测修复前 `hybrid_fusion_strategy="RRF"` 返回 `True`。同一处还把 `retrieval_method` 改为经 `require_retrieval_method` 收敛——与字面量比较对未知取值不会报错，而是静默判定「不是混合检索」。生产路径（`retrieve`）传的是收敛后的值，因此这条只在库调用方直接调本方法时可达，属契约一致性修复。

- **补齐模型配置展示路径的两类收敛**：`ModelDetail.provider` 声明为 `str`，但 `SessionConfig.llm_configurations`／`rerank_configurations` 的 `_VALIDATORS` 明确不登记（「按原样写入，由使用方负责其内部结构」），嵌入方可以往映射里放任意取值。`display_chat_config` 在 `start_chat_session_async` 里于任何提示符**之前**被调用，于是以下每一条都让会话在「知识快照加载成功」之后立刻终止：

  - **非字符串 `provider` 抛裸异常**：`display_chat_config` 拿它拼 `f"{provider.lower()}_api_key"`。实测 `provider=["openai"]` 更早一步——它在 `active_providers = {…provider}` 这个**集合**里就抛 `TypeError: unhashable type: 'list'`；`provider=None`／`5` 则抛 `AttributeError: 'NoneType' object has no attribute 'lower'`。两处判据（集合成员与 `.lower()`）都要收敛。现抽出共享助手 `require_provider_name`，`RetrievalService._get_rerank_provider` 里原本内联的同一判据改为复用它——两处不再各写一份。判据只要求「是字符串」，**不**查 `_provider_map` 白名单：白名单是工厂的职责，在展示层提前套它会让「配置里有个尚未支持的 provider」从「启动时列出配置」变成「启动时崩溃」。
  - **同一张表的裸 f-string 插值**：`model_name`、`active_llm_configuration`、`active_rerank_configuration` 是位数超限整数时抛 `ValueError: Exceeds the limit`。而**同一个函数**的 `top_k`／`score_threshold`／权重早已走 `_render_value`，注释还明确写着「本函数在任何提示符之前被调用，会话在启动横幅这一步就终止」——判据只做了一部分单元格。现四处统一经 `_render_value`（降级成量级描述，正常取值显示形态不变）。
  - 顺带把该函数里五处 `logger.debug(f"…{取值}")` 改成惰性 `%s` 参数：f-string 在 DEBUG 关闭时也**已经求值**，裸插值对同一取值照样抛。

- **补齐配置编辑器动态上界的规模字段收敛**：`config_menu._top_k_ceiling` 与 `_multiplier_ceiling` 读取另一个可直写的规模字段来算动态上界。修复前实测「`retrieval_candidate_multiplier` 被直写成 `"abc"`、用户在菜单里选择『调整 Top K』」时，`max()` 抛裸 `TypeError: '>' not supported between instances of 'str' and 'int'`；`top_k` 被直写成 `None` 时对称复现。配置编辑器在聊天主循环的 `try` 之外，内层只接 `KeyboardInterrupt`／`EOFError`，裸 `TypeError` 会以 traceback 终止整个会话。`retrieve` 对同一字段早有收敛判据，现将它上移为 `require_scale_field`（`src/utils/config.py`），服务层与 UI 共用；UI 不反向依赖服务层私有函数。非正整数继续按既有读取语义收敛为 1，`bool`／非整数／超上界取值给出带字段名的 `ValueError`。

- **补齐 `markup_text` 的模板形态校验**：该函数对**值**参数早已用 `_render_value` 完整收敛（位数超限的整数降级成量级描述），对占位符数量不符与 NUL 字符也都抛 `ValueError`；唯独**模板**参数没有形态判据，而函数体第一行就对它做成员测试（`if _MARKUP_PLACEHOLDER in template`）。实测修复前非字符串模板抛裸 `TypeError`，消息还随形态各不相同——`5`／`None` 是 `argument of type 'int' is not iterable`，`list`／`tuple`／`dict` 是 `expected string or bytes-like object`，`bytes` 是 `a bytes-like object is required`。三种消息都不指向「模板写错了」，也不是 `ValueError`：同一个「入参形态非法」的处置，同一个函数里两种收场。现统一收敛成带取值的 `ValueError`。

  可达性说明：44 个调用点里 43 个模板是字符串字面量，剩下 1 个（`scripts/embed_knowledge_base.py` 的 `mask_api_key(api_key_value)`）两个分支都返回字面量，因此本仓自带流程走不到——修的是**公开函数**的契约一致性。

- **补齐 Google 与 Ark 响应顶层容器的形态守卫**：两个 provider 的响应容器此前只写 `field(x, "candidates", []) or []` / `field(x, "output", []) or []`——那只挡住 `None` 与空值，挡不住其它形态。**可达性两者不同**，取决于各 SDK 的响应构造方式（详见第十四轮条目里的对照表）：Ark 走 `construct_type`，**不执行 pydantic 校验**，因此线上可达；Google 走 `model_validate`，畸形响应在 SDK 层就被拒，因此那里的守卫只在测试替身下可达。原文把两者一并称为「实测走真实入口」是不准确的——测试打桩了 `_get_client`，恰好绕过了校验所在的那一层。

  - Google `candidates=5` → 裸 `TypeError: 'int' object is not iterable`；`candidates={"0": {}}` → 更晚一步，在 `candidates[0]` 处抛裸 `KeyError: 0`。本文件有 **6 处**直接迭代 `candidates`（`_response_text`、`_extract_refusal`、`_extract_result` 两处、`_stream_events_from_chunk`、`_stream_chunk_has_terminal_event`），现全部经新增的 `_require_candidates` 收敛。
  - Ark Responses `output=5` → 裸 `TypeError`；Ark Chat `message.tool_calls=5` → 同一个裸 `TypeError`。现经新增的 `_require_ark_list` 收敛（6 处调用点）。

  这些异常都不是 `RuntimeError`——而两个 provider 的错误分类（以及调用方的 `except`）按 `RuntimeError` 写。同一个形态在 `openai_compatible._raise_for_chat_response_error` 对 `choices` 早已有显式守卫，三个 provider 的收场必须一致。`candidates` / `output` 缺失（`None`）的既有语义保持不变：流式路径的正常中间态依赖它。

- **补齐 OpenAI 兼容渠道响应容器的形态守卫，并厘清同类守卫在三个 SDK 上的可达性差异**：`OpenAICompatibleProvider` 有 **8 处**直接迭代响应容器（`output`、`message.tool_calls`、流式 `delta.tool_calls`），此前每处都只写 `field(x, ..., []) or []`——那只挡住 `None` 与空值。实测走真实入口：Chat 路径 `message.tool_calls=5` 与 Responses 路径 `output=5` 都抛裸 `TypeError: 'int' object is not iterable`，而调用方的错误分类按 `RuntimeError` 写。现新增 `_require_response_list`，判据与同文件 `_raise_for_chat_response_error` 对 `choices` 的守卫一致，8 处全部接入。这一条影响 `qwen`／`siliconflow`／`ollama`／`lm-studio`／`deepseek`／`grok` 等 7 个薄子类共用的路径。

  **同时厘清一个此前被我说错的事实**：同一类守卫在三个 SDK 上**可达性不同**，因为响应构造方式不同——

  | SDK | 响应构造 | 是否校验 |
  |---|---|---|
  | `google-genai` | `_from_response` → `cls.model_validate(response)` | **校验**，畸形容器在 SDK 层就被拒 |
  | `openai` | `_process_response_data` → `construct_type`（`_strict_response_validation` 实测默认 `False`） | **不校验** |
  | `volcenginesdkarkruntime` | 同样只有 `construct_type` | **不校验** |

  因此上一轮（第十三轮）提交里的 Google 守卫**只在测试替身下可达**——那批用例打桩了 `_get_client`，恰好绕过了校验所在的那一层，而原文把 Google 与 Ark 一并称为「实测走真实入口」是不准确的。Ark 的那批是线上可达的。Google 守卫本身保留（契约一致性有价值，且它暴露了一个不真实的既有夹具），但定位已在代码注释、测试 docstring 与本条目里改正；另加两条**前提断言**（`test_google_sdk_validates_response_containers_so_the_guard_is_not_reachable_live` 与 `test_openai_sdk_does_not_validate_response_containers_by_default`）把这个差异固定在测试里——某个 SDK 版本改了构造方式就会变红。

- **补齐 `rerank_enabled` 读取侧的判据（此前只有写入侧）**：`SessionConfig._validate_rerank_enabled` 只认 `bool`（`tests/runtime/test_contracts.py` 里就有 `("rerank_enabled", "yes")` 这一行），但那只管**写入**。`SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫，直写 `"false"`／`"no"`／`0` 之后，UI 的 **5 个读取点**走的是真值判断，与服务层的 `_coerce_rerank_enabled` 分成两套判据：

  - `display_chat_config`（`chat_table` 的状态格、Rerank 模型段落、API Key 段落的三处分支）：实测直写 `"false"` 时配置表显示 **[bold green]「启用」** 并渲染出激活的 Rerank 模型，而 `RetrievalService.retrieve` 对同一个值直接抛 `ValueError: rerank_enabled 必须是布尔值`。把 rerank 写成 `"false"` 的人看到的是「已启用」，且界面不提示任何错误——**与配置意图相反**。
  - `config_menu` 的菜单项与 `questionary.confirm(default=...)`：同一取值下菜单显示「当前: 启用」，且 `confirm` 在用户直接回车时把 `default` **原样**返回（`prompt_toolkit` 的 `set_answer` 分支），于是回车得到字符串 `"false"` 并被写回配置——界面显示「是」，写进去的是字符串，服务层再报错。

  现新增共享助手 `require_rerank_enabled`（`src/utils/config.py`），与 `_validate_rerank_enabled`／`_coerce_rerank_enabled` 共用同一判据（只认 `bool`，**不**接受 `0`／`1`／`"yes"`——真值语义正是这条缺陷的根因）。服务层的 `_coerce_rerank_enabled` 改为复用它，两处不再各写一份消息。

  **两处读取点的处置不同，这是有意的**：`display_chat_config` 与同一张表的 `retrieval_method` 一致，**直接拒绝**（它本就在启动横幅处对非法取值报错）；`config_menu` 则降级——菜单由 `launch_config_editor` 经 `asyncio.to_thread` 调用，调用点只接 `(KeyboardInterrupt, EOFError)`，抛 `ValueError` 会终止整个会话，而菜单是用户唯一还能改配置的入口（`display_chat_config` 在上游已经拒绝）。因此菜单里非法取值显示「未知」、`confirm` 的默认项传 `None`（questionary 的合法默认，不预选），用户仍能进菜单把它改回布尔。

- **修正 `top_k` 在同一个函数里被重读**：`RetrievalService.retrieve` 开头就把 `session_config.top_k` 收敛进局部变量（`require_scale_field`），但两处下游**重新读原属性**：

  - `retrieve` 出口的切片 `ranked_results[: max(1, session_config.top_k)]`；
  - `_rerank_if_needed` 的 `max(1, min(session_config.top_k, ...))`——该方法的 docstring 明确写着「供库调用方直接调本方法」，而库调用方**不经过** `retrieve` 开头那道收敛。

  该字段可被直写（`SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫）。静态非法值会被开头那道收敛拦下，但 `retrieve` 是 async、中间有多处 `await`——**检索飞行中**改写属性就绕过了它。实测两条路径都抛裸 `TypeError: '>' not supported between instances of 'str' and 'int'`（库调用方路径是 `'<'`），而调用方的错误分类按 `ValueError` 写。

  修法与同函数里 `rerank_enabled` 的既有处置一致：出口切片改用**已收敛的局部变量**；`_rerank_if_needed` 新增 `top_k` 形参，由 `retrieve` 传入收敛后的值，`None` 时自行经 `require_scale_field` 收敛（供库调用方直接调用的场景）。两处都只保留一份判据。

- **补齐 `display_utils._require_model_detail` 的不可哈希键守卫**：该函数的 docstring 明确以「同一个 key 走 `ModelProviderFactory._get_model_detail` 得到 `ValueError`，错误分类本就该一致」为理由建立契约，但工厂侧有一条它没有的守卫——**成员判断前先收敛键形态**。`key not in mapping` 对**不可哈希**的取值抛裸 `TypeError: unhashable type: 'list'`，而本函数在 `start_chat_session_async` 里于任何提示符**之前**被调用：实测直写 `active_rerank_configuration = ["siliconflow"]` 时，用户刚看完启动横幅就收到裸 traceback，会话随即终止。可达路径与其它 `active_*` 字段同源（`Settings` 与 `SessionConfig` 都没有为它们登记 validator，普通属性也没有 `__setattr__` 守卫）。

  守卫**只收不可哈希这一类，不要求 key 是字符串**。初版写成「必须是字符串」，被既有用例 `test_display_chat_config_renders_values_beyond_the_conversion_limit` 挡下：该用例明确要求位数超限的**整数键**可渲染（可哈希的非字符串键在映射里能正常取到值，那是显示层的既有契约）。工厂侧要求字符串是因为它要拿 key 去建 provider，两处的接受面本就不同；一致的是**异常分类**。

- **补齐退出钩子的系统目录清单**：`cleanup_temp_files` 是 `atexit` 钩子，删除前用 `_reject_unsafe_cache_dir` 判断 `cache_path` 是否可安全 `rmtree`。清单原先只有 `/etc`／`/usr`／`/bin`／`/sbin`／`/var` 与三个 macOS 目录，**漏掉了同样属于系统级、且删除后果不可逆的一批**。实测以下路径全部被放行，退出钩子会把它们整个删掉：

  - `/tmp`（在 macOS 上解析成 `/private/tmp`）——整个系统临时目录；
  - `/Users`——macOS 上**全部用户家目录的父目录**，`rmtree` 等于抹掉机器上每个账户的家目录；`/home` 在 Linux 上同理；
  - `/opt`——本模块自己的 docstring 就把「安装在 `/opt`」列为场景，而它的父目录同样不该被删；
  - `/dev`（设备节点）、`/proc`、`/sys`（内核虚拟文件系统）等。

  触发条件只是一个配置项：`config.toml` 里写 `cache_path = "/tmp"`（或 `/Users`／`/opt`）。判据补为**相等**，因此只挡系统根本身，卷内／目录内的正常缓存位置不受影响——`/tmp/myapp/cache`、`/opt/myapp/cache`、`/Volumes/MyDrive/cache`、`/Users/someone/Library/Caches/app` 全部继续放行（本函数从不过度拦截用户有意指定的缓存目录，见下方「目录名不是 `.cache` 时只警告不阻断」的既有取舍）。

- **补齐嵌套响应容器的形态守卫**：第十三／十四轮把**顶层**容器收敛了（Google 的 `candidates`、Ark 的 `output`／`tool_calls`、OpenAI 兼容的 `output`／`tool_calls`／`choices`），但**嵌套一层**的容器仍是 `field(x, ..., []) or []`——只挡 `None` 与空值，挡不住其它形态。实测全部抛裸 `TypeError: 'int' object is not iterable`，而各 provider 的错误分类按 `RuntimeError` 写。这是「判据只做了一半」的又一实例：同一个形态在顶层被拒、在下一层被放行。

  - **Google**（5 处 `candidates[].content.parts`、4 处 Embedding `embeddings`）：新增 `_require_parts` 与 `_require_embeddings`。可达性同 `_require_candidates`——google-genai 走 `model_validate`，畸形容器在 SDK 层就被拒，因此守的是替身场景与契约一致性。
  - **Ark**（3 处 `output[].content`、2 处 tokenization `data`）：新增 `_require_ark_content`，`data` 复用既有的 `_require_ark_list`。**线上可达**（Ark SDK 走 `construct_type`，不校验）。
  - **Anthropic**（顶层 `content`）：**线上可达**——实测 `anthropic` SDK 的 `_strict_response_validation` 默认 `False`，与 openai／Ark 同类。
  - **OpenAI 兼容**（3 处 `output[].content`，以及 Responses 路径的 `choices`）：`choices` 这条尤其要紧——**Responses 路径没有别处替它把关**：`_raise_for_chat_response_error`（含 choices 形态守卫）只被 Chat 路径调用，而 `_invoke_responses` 走的是 `_raise_for_response_error`，后者只查 `error` 与 `status`。实测 Responses 响应带 `choices=5` 时抛裸 `TypeError`。

  共 19 处读取点接入守卫；`None` 与空值的既有语义保持不变（`None` 继续表示「该字段不存在」，正常中间态依赖它）。

### 测试

- 新增第十九轮的回归用例（20 个用例；回退源码后其中 21 个参数化实例变红）：Google 嵌套 `parts`／`embeddings`、Ark 嵌套 `content`／tokenization `data`、OpenAI 兼容 Responses `choices` 与嵌套 `content`、Anthropic 顶层 `content` 六类读取点各对 `int`／`mapping`／`str` 三种非法形态的拒绝；以及三条「不误伤」反向守卫——Google `parts`、Ark 嵌套 `content`、Anthropic `content` 的正常响应照旧解析。

- 新增第十八轮的回归用例（20 个用例；回退源码后其中 16 个变红）：16 个系统级根（`/tmp`／`/private/tmp`／`/Users`／`/home`／`/opt`／`/dev`／`/proc`／`/sys`／`/boot`／`/run`／`/srv`／`/mnt`／`/media`／`/root`／`/Volumes`／`/private`）必须拒绝；以及 4 个「不误伤」反向守卫——系统根**内部**的缓存目录（`/tmp/myapp/cache`／`/opt/myapp/cache`／`/Volumes/MyDrive/cache`／`/Users/someone/Library/Caches/app`）照常放行。

- 新增第十七轮的回归用例（12 个用例；回退源码后其中 4 个变红）：三类不可哈希 key（`[1]`／`{"a": 1}`／`{1, 2}`）必须收敛成带指引的 `ValueError`；同一个 key 在展示层与工厂侧必须给出同一种异常分类；以及四组「不误伤」反向守卫——可哈希的缺失键（`"missing"`／`5`／`None`／超大整数）仍是原来的「未找到key」消息、位数超限的整数键仍能取到值、既有的渲染用例保持绿。

- 新增第十六轮的回归用例（5 个用例，回退源码后全部变红）：库调用方直调 `_rerank_if_needed` 时四类非法 `top_k`（`"abc"`／`None`／`[1]`／`True`）必须收敛成带指引的 `ValueError`；以及检索飞行中把 `top_k` 直写成 `"abc"` 后，出口切片仍须正常返回（用已收敛的局部变量）。

- 新增第十五轮的回归用例（10 个用例；回退源码后其中 9 个变红，1 个「不误伤」反向守卫保持绿）：六个非布尔取值（`"false"`／`"no"`／`"true"`／`0`／`1`／`None`）下 UI 与服务层必须给出同一种收场；配置表不得把 `"false"` 显示成「启用」（原始症状）；菜单的标签与 `confirm` 默认项对非法取值降级为「未知」／`None`；以及真布尔取值照旧渲染「启用」／「禁用」的反向守卫。

- 新增第十四轮的回归用例（8 个用例；回退源码后其中 5 个变红）：OpenAI 兼容渠道 Responses `output` 对三类非序列取值的拒绝、Chat `message.tool_calls` 的拒绝、流式 `delta.tool_calls` 的拒绝（走同一守卫）、正常响应（含工具调用）照旧解析；以及两条**可达性前提断言**——google-genai 在 SDK 层就拒绝畸形容器、openai 默认不校验。后两条在回退源码后仍为绿（它们断言的是 SDK 行为而非本仓行为），这正是它们的作用：把「哪个守卫线上可达」这件事从口头结论变成会随依赖升级而失效的测试。

- 新增第十三轮的回归用例（18 个用例；回退源码后其中 7 个变红，11 个「不误伤」反向守卫保持绿）：Google `complete` 对三类非序列 `candidates`（`int`／`mapping`／`str`）的拒绝、流式两个读取点用同一守卫、`candidates=None` 的既有语义不变、正常响应照旧解析；Ark 两条协议分别对非序列 `output` 与 `tool_calls` 的拒绝，以及正常响应（含工具调用）照旧解析。用例走 `complete` 入口而不是直接调私有解析函数——「守卫加在哪个函数上」与「路径是否经过它」是两回事。但要注意**可达性两半不同**：Ark 的用例线上可达，Google 的打桩了 `_get_client`、绕过了 SDK 的 `model_validate`，因此只在替身场景下成立（第十四轮已把这条前提写成断言）。

  同时修正 `tests/providers/test_google_genai.py` 的一处夹具：它用未指定 `candidates` 的裸 `MagicMock`，而真 SDK 的字段类型是 `list[Candidate] | None`。裸 `MagicMock` 的属性是一个可迭代的空 mock，旧代码只做 `or []` 与迭代，这种不真实形态恰好能通过；形态守卫会把它拒掉。夹具已写成真实形状（显式 `candidates = []`）。

- 新增第十二轮的回归用例（12 个用例；回退源码后其中 7 个变红，5 个「不误伤」反向守卫保持绿）：`markup_text` 对七类非字符串模板（`int`／`None`／`list`／`tuple`／`dict`／`bytes`／位数超限整数）的拒绝、四类合法模板（含零占位符与多占位符）照旧渲染，以及「形态守卫不得盖掉既有占位符判据」——模板是合法字符串但数量不符时仍须给出原来那条消息。

- 新增第十一轮的回归用例（16 个用例；回退源码后其中 11 个变红，5 个「不误伤」反向守卫保持绿）：两个动态上界助手对字符串／`None`／列表／`bool`／位数超限整数的拒绝、两种方向的合法上界照旧工作、真实 `edit_retrieval_params` 入口在坏倍率下给出可读错误而不是终止，以及 `require_scale_field` 与服务层既有的 `max(1, ...)` 读取语义对齐。

- 新增第十轮的回归用例（15 个用例；回退源码后其中 9 个变红，6 个「不误伤」反向守卫保持绿）：`display_chat_config` 对四类非字符串 provider（`list`／`dict`／`None`／`int`）的拒绝、Rerank 侧同一判据、三类位数超限取值（`model_name`／两个 `active_*_configuration`）**必须可渲染**而不是终止会话、合法配置照旧渲染，以及共享助手 `require_provider_name` 接受任意字符串（含尚未支持的 provider）并拒绝其余形态。位数超限那两条特意把激活名与配置键一并设成同一取值——它们同源，只设激活名会先撞上「未找到key」那条判据，测不到渲染路径。

- 新增第九轮的回归用例（18 个用例；回退源码后其中 11 个变红，7 个「不误伤」反向守卫保持绿）：`generate_reply` 对六类不可用 `chat_temperature`（`str`／`None`／`list`／`bool`／负数／位数超限整数）的拒绝**且断言被拒前未发起请求**（只断言「抛了 ValueError」会让「先发请求再校验」的实现也通过）、五条合法取值照旧传给模型并收敛成 `float`、配置边界与运行期读取侧对同一取值给出同一种收场（这条盯「只修一边」）；以及 RRF 融合在任何大小写／空白形态下都跳过阈值过滤、`weighted` 与非混合检索仍套用阈值、兜底分支对无法识别的策略报错而不是静默当成「非 RRF」。

- 新增第八轮的回归用例（19 个用例；回退源码后其中 12 个变红，7 个「不误伤」反向守卫保持绿）：`retrieve` 对六类不可用 `retrieval_method`（英文别名／缩写／`None`／`int`／`list`／位数超限整数）的拒绝**且断言被拒前未发起任何一路检索**（只断言「抛了 ValueError」会让「先检索再校验」的实现也通过）、`display_chat_config` 对四类的拒绝、六条合法取值（枚举成员与字符串取值 × 三种模式）照旧走对分支、合法取值照旧渲染、配置边界与运行期读取侧对同一取值给出同一种收场（这条盯「只修一边」），以及共享助手对鸭子类型替身的接受与对取值形态不对时的拒绝。

- 新增第七轮的回归用例（14 个用例；回退源码后其中 10 个变红，4 个「不误伤」反向守卫保持绿）：`SessionConfig.set_many` 对五类非映射容器（`int`／`None`／`str`／`bytes`／`list`）的拒绝、`Chatbot.chat_config` setter 对同样五类的拒绝、两条合法路径照旧工作（映射与 `SessionConfig` 实例），以及被拒时状态不被改了一半。

- 新增第六轮的回归用例（22 个用例；回退源码后其中 18 个变红，4 个「不误伤」反向守卫保持绿）：``Pipeline.process`` 对四类非法入参的拒绝、``BasicCleaner.clean`` 对四类非法容器与三类非法条目的拒绝、``RecursiveTextSplitter.split`` 的同两组、``split_hierarchical`` 与 ``split`` 共用同一判据（这条专门盯「只守卫一个入口」），以及三个阶段合法路径照旧工作（含空列表与清洗结果）。


- 新增第五轮的回归用例（9 个测试函数，覆盖 24 条 provider×方向路径；回退源码后其中 8 个变红，1 个「不误伤」反向守卫保持绿）：四个 provider 的工具转换器对不可迭代取值的拒绝与对 `None`／空列表／合法工具的接受、`extract_system_prompt` 对四类非法容器与两类非法条目的拒绝、两个 rerank provider 同步与异步入口对非序列 `documents` 的拒绝、四个 embedding provider 同步与异步入口对四类非法 `texts` 的拒绝（含 `"abc"` 这条静默拆分的）、以及 `require_text_list` 报错消息带下标。


- 新增 Anthropic `user`／`betas` 取值的回归用例（14 个；回退源码后其中 12 个变红）：`user` 的四类非字符串形态、`betas` 的四类非列表形态（`int`／`bytes`／裸字符串／`Mapping`）、`betas` 元素的三类非字符串形态，以及三条「不误伤」的反向守卫（合法 `user` 与 `betas` 仍合并进 `anthropic-beta`、`betas` 单独出现时原样进 params 不注入 profile 头）。另有 `test_anthropic_betas_validated_without_user` 专门盯住「校验只加在 `_user_profile_fields` 里」这条漏网路径——`betas` 在 `user=None` 时不经过那个函数。测试的捕获 helper 同时装上 `create` 与 `count_tokens`，因为 `betas` 只被前者转发（`beta_count_tokens` 的白名单放行它但函数体不读）。



- 新增第三轮修复的回归用例（`tests/test_boundary_hardening_regressions.py` 第六节，16 个用例；回退源码后其中 13 个变红）：`extra_headers` 的四类非字符串取值（`int`／`bytes`／`list`／`dict`）与 `None` 值剔除、`user` 已设置而 `extra_headers` 未提供、beta 计数入口同步与异步两条路径都用校验后的头、合法头（含既有 `anthropic-beta` 值）照旧保留并合并；以及 `embedding_model_detail` 对五类不可用取值（`None`／字符串／整数／缺 `model_name`／缺 `provider`）的拒绝与两条不误伤的接受路径，另有一条覆盖**零分块**建库路径（它跳过 embedding 调用、最晚暴露）。三条未变红的是「收紧不误伤」的反向守卫。beta 计数入口的用例整体替换 `_get_client`／`_get_aclient`（客户端是懒加载的，构造期 `_client` 为 `None`），沿用仓库既有协议适配测试的注入方式。

- 新增本轮全部修复的回归用例，逐项经红绿验证（回退源码后对应用例变红）：发布工作流两条 6000 条目的大归档用例（分别盯住「凭证检查漏报」与「`tiktoken_cache` 误报」两个方向）、`get_relative_path` 的相对形态与超长路径不被截断、Excel 日志五类 XML 敌对码位的 `load_workbook` 回读（外加一条合法码位不被误伤的反向守卫与一条区分「XML 规范化 `\r`」和「被替换成 `\ufffd`」的用例）、嵌入 provider key 缺失时的 `ValueError`、四种畸形 `pyproject.toml` 形态的子进程降级、清洗器哨兵的一次扫描（断言复杂度量级而非绝对耗时）与边界无歧义、分片器的 `mode`／尺寸参数／分隔符长度三类构造期拒绝、`SessionConfig` 的字段名与属性直写兄弟值、清单写入侧的文本与计数字段、以及 `/config` 被拒后仍在会话内（含一条断言被拒消息经脱敏）。
- 修复四处「修复没有测试覆盖」的假绿：对本轮九项修复做定向变异（逐项把源码改回修复前的形态，再跑对应测试），其中四项回退后测试**仍然全绿**——BM25 的「所有 token 都为空」守卫、`_parent_key` 的 `str()` 替换、`parents.pkl` 错误消息的 `safe_repr`、以及 `retrieval_candidate_multiplier` 的 `inf` 分支。前三项现补上直接触发该分支的用例（`[[]]`／`[[], []]`／混合形态、`pickle` 往返的 `parent_id = 10**4301`、`parents.pkl` 写入 `{10**4301: [1, 2]}`）。第四项暴露的是**两处独立防线**：`isfinite` 守卫（接住 `config.toml` 的 `inf`）与 `int()` 的 `OverflowError` 兜底（接住 `Decimal('Infinity')` 这类非 `float` 的非有限值），只变异其中一处时另一处仍然生效、测试自然全绿——现为两条路径各补一条用例并分别变异验证。回退后这四处全部变红。

- 修复 `test_dotenv_is_resolved_from_root_dir_not_cwd` 的假绿：原实现直接改写 `Settings.model_config`，断言比较的是测试自己注入的值，把 `env_file` 回退成相对路径后依然通过。现改用子进程伪造冻结环境，让 `Settings` 在类定义期走真实求值路径，回退修复时该测试会失败。
- 修复发布脚本测试对构建机缓存与网络的隐式依赖：`prepare_runtime_layout` 与 `stage_bundle` 会触发 BPE 预置，其源路径取决于宿主机 `$TMPDIR` 或联网。现把预置从布局准备中拆出，并为需要它的用例提供自给夹具，使 `tests/test_release_scripts.py` 在干净 `TMPDIR` 且断网时仍全绿。
- 新增 BPE 预置的正反用例：哈希正确时按 `tiktoken` 的查找键名落盘，内容损坏时拒绝打包。
- 新增版本一致性与日志降级用例：产物自报版本必须与 `pyproject.toml` 一致（构建脚本据此断言），降级 warning 只发一次。
- 新增冻结环境的分词缓存用例：包内缓存存在时 `TIKTOKEN_CACHE_DIR` 必须被指向它——即使用户环境里已有一个空串值（`tiktoken` 把空串视为禁用缓存），以及缺失时必须有警告。两条都走子进程伪造 `sys.frozen`，因为该分支在 import 期只求值一次，`monkeypatch` 触达不到。
- 新增发布工作流的 shell 逻辑回归测试（`tests/test_release_workflow.py`）：把步骤的 `run` 脚本从 YAML 里取出来，用合成归档在本地真跑一遍，覆盖凭证文件、缺失 `tiktoken_cache`、空归档集、`download-artifact` 的嵌套布局、zip 分支，以及校验和的落盘位置与「用户视角 `shasum -c` 能通过」。旧实现的单层 glob 会在这组用例下全部变红。
- 新增退出清理的安全边界用例：`cache_path` 指向应用根目录（`.`、`./`、`data/..`、`..`）或应用运行时目录（`data`、`knowledge_base`、`data/logs`、`data/kb`）时必须拒绝删除，且端到端断言 `cleanup_temp_files()` 不会真的删掉 `.env` 与源文档；同时断言普通缓存目录（`.cache`、`data/cache`）不被误拒。
- 新增日志清理的作用域用例：只删本应用命名的日志，`nginx_access_*.log`、`postgres_*.log`、`my_app_log_*.log` 等一律保留，未过期的本应用日志保留，非 `.log` 文件不参与。
- 新增终端展示的凭证边界用例：`display_chat_config` 渲染出的文本不得含 API Key 或 Base URL 中内嵌的凭证，同时主机名必须保留（脱敏不能把 URL 整个抹掉）。
- 新增活动快照标记文件的取值用例：`../` 逃逸、绝对路径、含分隔符与 `..` 的取值一律拒绝，本应用自己生成的 id 形态照常接受。
- 新增快照 id 逃逸用例：`../outside`、`../../etc`、绝对路径、`a/../..`、`..` 在四个拼路径的方法上逐一断言拒绝，并端到端断言快照根目录之外的目录未被删除；同时断言生成 id 的正常流程照常走通。
- 新增混合合并的键碰撞用例：前 64 字符相同的两篇文档必须各占一条，同一篇文档在两路结果里出现时仍须合并计分。
- 新增权重边界的构造器用例：`nan`／`inf`／`-inf`／负数／大于 1 在 `SessionConfig(...)` 与 `HybridReranker` 上逐一断言拒绝，合法权重照常接受。
- 新增构造器与 `__setitem__` 的一致性用例：对 11 组非法取值断言两条入口都拒绝，合法取值在两条入口上得到相同的配置。这类「只修了一条入口」的问题此前已发生过一次（`top_k` 上界），逐字段比对能防止后续新增字段时再漏一侧。
- 新增候选规模上界的用例：`Settings` 与 `SessionConfig` 都必须拒绝「两个字段各自合法、乘积越界」的组合，恰好等于上界的组合必须接受；并断言配置编辑器给出的上界配上另一字段的当前值一定不会被 `SessionConfig` 拒绝（这类不一致会以裸 traceback 终止会话）。
- 新增配置批量更新的原子性用例：任一字段非法时整批都不生效（原配置逐字段不变），合法批量更新整体生效。
- 新增索引文件行数的用例：`semantic.index` / `lexical.index` 与 `chunks.pkl` 条数不一致时必须拒绝，一致时通过。同时修掉两处快照夹具：它们用占位字节写 `semantic.index`，而真实加载路径会 `faiss.read_index` 它——这类夹具构造出的快照本来就不可能被加载，等于绕过了被测行为。
- 修掉向量有限性用例泄漏的 `RuntimeWarning`：`np.array(..., dtype=np.float32)` 对超出 float32 范围的值只发 warning 后静默溢出，那条用例正是要覆盖这个行为，现在显式断言该 warning 而不是让它混进测试输出——泄漏的 warning 会掩盖真正的新警告。

- 新增基础清洗器的哨兵碰撞用例：正文里出现占位串字面量时必须原样保留，整段由该串构成的文档不得被抹成空串。
- 新增 `upsert_embeddings` 的失败原子性用例：维度不一致被拒后 `documents` 与 `embeddings` 行数必须仍然一致、`save_snapshot` 不会落盘不自洽状态；此前该方法没有任何直接测试覆盖。

- 新增富文本标记边界的回归用例：`markup_text` 对 10 类标记形态（`[/]`、`[/color]`、`[/if]`、`[b]…[/b]`、markdown 链接、`[url=…]`、反斜杠游程、尾部反斜杠）断言原样保留，同时断言模板自身的样式仍生效、值继承所在位置的样式、多占位符按序落位、模板与实参数量不符时显式报错。聊天、召回测试与 `main.py` 顶层错误出口各自新增端到端用例：文档正文／用户查询／异常消息含 `[/]` 时必须照常渲染（修复前这些用例全部变红）。红绿验证方式为把调用点回退成 f-string 拼接，三个路径各自复现 `MarkupError`。
- 新增 `parents.pkl` 的校验用例：无法解析、顶层非映射（`None`／`int`／list／str）、父条目既非 dict 也非 str 四类必须拒绝，正常的 dict 与字符串形态照常通过。同时修掉 `tests/retrieval/vdb/test_factory.py` 里用空字节写 `parents.pkl` 的夹具——这类占位文件构造出的快照本来就不可能被加载，等于绕过了被测行为（与上一轮 `semantic.index` 的夹具同一问题）。

- 新增 Excel 日志的类型用例：以 `=` 开头的内容、查询与来源必须按文本落盘（断言 `data_type` 而不是只看 `value`——`load_workbook` 读回公式时 `value` 仍是原字符串，只看值无法区分），并断言内容逐字保留、不加前缀。修复前该用例在 `data_type` 上变红。

- 新增 manifest 的 TOML 往返用例：10 类取值（反斜杠、双引号、单引号、换行、制表符、控制字符、方括号、井号、等号、中文）在 `embedding_model` 上必须原样读回，并逐字段断言每个字符串字段都做了转义（不能只修被报告的那一个）；同时断言计数字段仍是 TOML 整数而非带引号的字符串。此前 `to_toml` 没有任何直接测试覆盖——两个夹具只是调用它来搭快照，值都是安全的常量。

- 新增 `config.toml` 语法错的用例：必须转成带 `配置解析失败` 前缀的 `ValueError`，且消息保留行号。第二条断言同时查前缀与行号——tomllib 的原始消息里本来就有 `line 2`，只查行号的话未包装的裸异常也能通过，测试会因为错误的原因变绿。

- 新增 embedding 维度的上界用例：`local_hash` 在构造期拒绝超界 `model_name`（断言异常发生在构造而不是 embed——只在 embed 里判的话向量已经建好了），上界以内含边界照常可用，无数字后缀时仍回落默认维度；`EmbeddingService._as_float32_matrix` 另按同一常量断言 65537/10⁶/10⁸ 被拒、1/256/3072/65536 被接受。

- 新增文档契约的载入期用例：`page_content` 为 `None`／整数／列表、`metadata` 为列表／字符串／整数、条目本身不是映射共 7 类必须拒绝，合规文档照常载入且 `metadata` 缺省仍补成空 dict。同时新增展示层与 prompt 构造的形态用例：三个提取助手逐形态断言退回默认值，两个展示函数与 `_build_prompt` 对 9 类异常条目必须照常完成，并断言非字符串内容**不会**以 `内容: None` 的形式进入 prompt。`RetrievalService` 另断言 store 返回 `None`／映射／字符串列表／含 `None` 的列表时给出可读的 `TypeError`，而 `metadata` 非映射时检索照常完成。

- 新增 Rerank 入参形态的用例：内容为 `None`／列表／整数时必须收敛成空串，正常内容原样传入。

- 新增轮转备份清理的用例：三个直接构造的备份名断言被清理、未过期的备份断言保留，另有一条**走真实 `RotatingFileHandler`** 让 handler 自己轮转出备份、再断言清理规则覆盖真实产生的文件名——直接写 `.log.1` 这样的名字等于假设了备份形态，而这条用例断言的是 handler 的实际行为（若 `doRollover` 改变命名，它会失败）。变异验证：把模式退回 `$` 结尾后这两条变红。
- 新增 PEM 脱敏的用例：六类私钥形态（RSA／通用／OpenSSH／EC／被截断无 END／嵌在异常消息里）断言正文不出现在脱敏结果里，并断言替换标记确实出现（只查正文不在的话，一个把整段文本删空的实现也能通过）；另有一条走真实链路——把 PEM 当 options 键名让 `validate_secret_free_options` 拒绝，再经 `safe_exception_text` 断言展示给用户的消息里不含正文。反向守卫覆盖证书、公钥与普通文本，防止把 `CERTIFICATE`／`PUBLIC KEY` 一起脱敏掉（它们不是凭证，抹掉会让用户失去上下文）。变异验证：去掉 PEM 规则后这 7 条全部变红。
- 新增 `HIDDEN_IMPORTS` 的一致性守卫（`tests/test_release_scripts.py`）：正向断言注册表里每个 provider 模块都在清单里，反向断言清单里没有已不在注册表中的模块名（只查一个方向的话，注册表删掉一个 provider 后旧名字会一直留着，而 PyInstaller 对不存在的 hidden import 只警告——清单会慢慢失真）。第三条用例校验前两条的**前提**仍然成立：逐个用 `grep` 确认这些模块确实没有静态导入路径，前提不成立时那两条守的是一个不存在的问题。另新增自检守卫的用例：注入一个 `ImportError` 后必须报成带 `HIDDEN_IMPORTS` 指引的 `RuntimeError`，而不是让 `ImportError` 穿出去。变异验证：从清单里删掉一个模块、以及往清单里加一个不存在的模块，两条守卫各自变红。
- 新增 `save_snapshot` 与 token 计数的用例：`metadata=None` 的条目必须能存快照，且断言与「metadata 缺键」这条既有路径**同值**（而不是断言某个具体数字——`document_count` 的口径是「distinct source」，没有 source 的条目会以 `None` 占一个桶；要紧的是两条路径得到同一结果）。`require_token_count` 对 11 类畸形取值（缺字段、非数值字符串、浮点字符串、位数超限的数字串、浮点数、`inf`、`nan`、列表、映射、布尔值、负数）断言报成带 provider 与字段名的 `RuntimeError`，并单列一条断言它**永不**抛出 `RuntimeError` 以外的类型——`int(float("inf"))` 抛的是 `OverflowError`，既不是 `ValueError` 也不是 `TypeError`，任何按后者写的处理都接不住。`Anthropic options.max_tokens` 对 `inf`／`-inf`／`nan`／`1e400` 断言得到「必须是正整数」消息，其余非法取值与合法取值一并覆盖。回退四处修复后这 20 条全部变红。
- 新增快照元数据字段取值的用例：`manifest.toml` 的 `chunk_count` 写成非数值字符串／位数超限的数字串／整个字段缺失，以及 `stats.json` 的 `chunk_count` 写成非数值字符串／浮点字符串／列表／位数超限的数字串，四类 `stats.json` 形态用参数化覆盖。每条都同时断言「有重建指引」与「保留了原始异常类型」——只查指引的话，一个把任意异常都包成同一句话的实现也能通过；只查类型的话，未包装的裸异常也能通过。另断言合法 manifest 照常通过（收紧不误伤）。回退两处修复后这 7 条全部变红。另新增整数解析判据的用例：

## [1.4.0] - 2026-09-20

### Provider 与协议

- 建立统一 Provider 工厂与抽象请求模型，补齐 LLM、Embedding、Rerank 的能力声明、配置校验、缓存与同步/异步生命周期管理。
- 新增 `complete()`/`acomplete()` 高级调用入口，统一支持多轮消息、工具调用、结构化输出、多模态内容、usage 与结束原因；`invoke()`/`ainvoke()` 保留文本流兼容行为。
- 修复多轮工具调用的历史归一化：共享的 `normalize_messages` 不再强制要求工具调用带 `id`，因为 Gemini 的 `FunctionCall.id` 是可选字段且 SDK 默认 `None`，否则把上一轮工具调用回填给 Google 会在请求发出前被本地拒绝；需要 `id` 的 Chat Completions 与 Responses 改为显式校验。同一处还修复了 `input` 参数别名被静默清空，以及 `function_call`/`custom_tool_call` 类型未映射为 `function` 的问题。
- 支持 Chat Completions、Responses、Google Generate Content 和 Anthropic Messages 四类线协议；OpenAI 兼容渠道与 Ark 可通过 `protocol = "responses"` 选择 Responses，Ark 的 `ark` 取值是 Chat Completions 的别名。
- Responses 对非官方端点要求 `options.server_verified_protocols = ["responses"]` 显式登记；未登记时工厂、Provider 和资源 Facade 均在请求前拒绝，避免把本地 SDK 资源误发到未实现 `/responses` 的服务端。
- Ark Responses 在构造阶段拒绝 `instructions` 与 `caching={"type": "enabled"}` 同时出现：官方规定配置 `instructions` 后本轮请求无法写入或使用缓存，`caching` 为 `enabled` 时服务端直接报错，而 SDK 不做本地校验。错误信息会指出 `instructions` 来自 `system_prompt` 的兼容默认值，并给出改走 `messages` 的修复方式。
- 新增原生资源 Facade，显式暴露文件、批处理、缓存、向量库、token 计数、调优等 SDK 能力，不再把资源生命周期混入普通聊天请求。
- 深度适配 Google GenAI、Anthropic、Volcengine Ark、Jina 与 SiliconFlow Rerank；OpenAI 兼容渠道复用 `openai` SDK，Jina 与 SiliconFlow Rerank 使用 HTTP JSON。
- 收紧动态资源树的能力门禁：此前按路径分段判断时只认复数 `responses`，而 `input_items`（实际请求 `/responses/{id}/input_items`）不含该分段，在未登记 `server_verified_protocols` 的渠道上也能把请求发到远端；同时动态路径只做能力检查、不执行 Facade 的参数校验，可带着互斥参数直接发出。现在两条路径共用同一套校验，`extra_body` 里的 `instructions`/`caching` 同样参与互斥检查。
- 顶层 Responses item 形态（`{"type": "input_audio", ...}`）补齐变体校验：此前只在缺少 `content` 键时检查，加一个无关的 `content` 键即可改走消息分支绕过全部校验；现在按 `type` 分流并复用同一套内容块实现。
- 修复 Responses 流式工具调用的 `id` 语义：`id` 此前在 delta 事件取输出项 ID、在 done 事件取调用 ID、合并时又互相覆盖，同一轮工具调用在 delta 与 completed 事件里得到不同的值，调用方按 `tool_call["id"]` 回填 `role=tool` 时会配不上。现在与非流式路径统一取调用标识符。
- 放宽 Ark Responses 的失败判定：内置工具调用项（`web_search_call`、`mcp_call`、`mcp_list_tools`、`image_process`、`agent_tool_call` 等）都是 SDK 输出项联合的正式成员，且都是需要后续轮次的中间态，此前被误判为「响应结构与预期不符」而无法处理。空摘要的 `reasoning` 不在豁免之列——它不是工具调用，豁免会让「completed 但什么都没有」静默返回空成功。
- 动态资源路径的能力门禁补齐：兼容渠道的路径映射此前只覆盖 responses/files，其余资源树（batches、vector_stores、images 等）拿到空能力直接放行，而显式 Facade 名会被拦；Ark 的动态路径按动词白名单决定是否做参数校验，漏掉 `async_create` 这类写法。两者都改为按资源分段与参数内容判断。判断依据用「参数里是否出现受校验字段」而不是动词，`extra_body` 不在此列——它是 retrieve/delete/list 的合法参数，放进触发集会把只读操作误判为创建请求、要求提供 `model` 与 `input`，使 Facade 可用而动态路径不可用。
- 资源代理不再经私有名转发到底层 SDK：`proxy.__dict__["_client"]` 此前能拿到未包装的原始客户端，绕开全部凭证扫描与能力门禁；`dir()` 也把这些通路提示给使用者。出口只保留文档化的 `.native`。
- 修正动态资源路径的能力映射与官方声明集不一致：SDK 客户端的属性是复数 `moderations`，段映射只写单数会让 `resources.moderations.create` 拿到空能力直接放行，而 `create_moderation` 会被拦；同时段映射列出了 `skills`/`realtime`/`webhooks`/`admin`/`content_provenance_checks` 五项官方声明集之外的能力名，把官方端点从放行变成显式报错——而这些资源在 `capabilities` 里声明为可用。段映射现在限定在声明集内。
- Google 的实验性资源门禁此前只认显式 Facade 名：动态路径是点分形式（`google.interactions.create`），资源名是独立分段，按前缀匹配全部落空、直接放行——用户拿一个不含该资源的客户端走 `resources.interactions` 就能绕开。改为同时按路径分段匹配。
- Embedding 区分文档与查询任务类型：Google 使用 `RETRIEVAL_DOCUMENT` 与 `RETRIEVAL_QUERY`。

### 安全

- 新增配置、请求与日志边界的凭证校验：模型级 `options` 与请求级扩展不得携带密钥、请求头、query 或 Base URL 覆盖，资源 Facade 的关键字与位置参数同样受检。
- 凭证键识别补齐火山引擎的 `ak`/`sk` 与 Azure 存储的 `account_key`；此前这些键可绕过请求覆盖校验，而 `api_key` 会被拒绝。
- 日志与错误信息统一脱敏常见凭证表示（Bearer、API key、URL userinfo 与 query）；补齐 `ak=`/`sk=`/`account_key=` 形式，以及服务端错误里「首尾可见、中间掩码」的凭证形态（如 `sk-abc***...***xyz`，此前会原样落进日志）。
- 修复脱敏日志 formatter 的缓存泄漏：`logging.Formatter` 会把格式化后的 traceback 缓存在 `record.exc_text` 上供后续 handler 复用，先格式化后脱敏会让同一 logger 上的非脱敏 handler 输出原始凭证；现在脱敏结果会写回缓存。
- 文本脱敏的键名识别与配置边界对齐：此前词表更窄，且键名前要求非字母数字字符，`dbPassword`、`myApiKey` 这类驼峰键在配置边界被拒绝、在日志里却明文输出。同时给值加上形态约束，避免 `auth: none`、`cookie: enabled` 这类普通文本被误判为凭证而让合法 options 在边界被拒。
- 词边界改用「非 ASCII 字母数字」而非 `\b`：中文属 `\w`，`\b` 在 `密钥sk-...` 两侧都不成立，国产网关（SiliconFlow、Ark、DashScope）的中文错误消息此前会让密钥整串落进终端与日志。
- 键名识别改用「非字母数字或 camelCase 边界」：既让 `dbPassword`、`myApiKey` 这类驼峰键脱敏，又不把 `topsecret`、`sessiontoken` 从词中间切开（配置边界按 camelCase 切词判它们非敏感，切开会导致合法配置被误拒）。排除误判源改用非凭证字面量枚举，而不是值的长度/字符构成——后者会把 `password: FakePwOnly` 这类短纯字母凭证一并放过。
- 修复掩码规则的二次回溯：尾部前瞻逐字符重复扫描剩余串，在尾部无冒号时退化成 O(n²)（4000 字符的掩码密钥前缀加长 token 耗时 1.4 秒）。该函数挂在每条日志的 formatter 上，一个回显错误体的网关响应即可拖住进程。
- 掩码规则不再吞掉键值对的键名后留下孤儿值：`sk-abc***xxxtoken=<secret>` 里的键名不是敏感键（按 camelCase 切词判非敏感），键值规则不会接手，而掩码又已把键名吃掉，值失去锚点后明文落进日志。现在掩码匹配把键值尾部一并纳入，只保留键名作为可读上下文。
- 文本脱敏的键名判定改为调用 `is_sensitive_option_key`，与配置边界共用同一个函数：此前文本侧用正则的字符断言去逼近切词逻辑，`passwordHash`、`tokenCount`、`cookieJar` 这类「凭证词 + 另一个词」的键两个方向都不一致——真凭证在日志里明文输出，而键名相同的合法配置值又在边界被误拒。一致性现在由构造保证，不再依赖两条独立实现碰巧对齐。
- 短凭证键（`ak`/`sk`）的值补上非凭证字面量排除：这两个键在配置里常作开关（`sk: off`），与裸关键词（`auth: none`）是同一类误判源，此前只给裸关键词加了约束。
- 脱敏替换不再改写分隔符：此前把 `key: value` 归一化成 `key=value`，会篡改被脱敏文本的结构（JSON 片段因此变成非法 JSON）。现在保留原文的 `:`/`=` 与引号。
- 基类 `_require_provider_resource` 的 `kwargs` 契约写明当前实现状态：只有 Volcengine 在 Facade 上做参数校验、因而需要 `kwargs`，其余 Provider 忽略它与契约一致；同时注明凭证边界不依赖本钩子，代理层在调用门禁前已统一校验。
- 修复替换函数里的二次回溯：尾部游程查找用无锚点的 `re.search(r"[A-Za-z0-9_-]+$")`，在每个起点重试 `+$`；尾部以 `.` 这类掩码占位符收尾时游程够不到 `$`，退化成 O(n²)（32010 字符 9.6 秒，4010 字符 86ms）。仓库原有线性测试的用例恰好无尾随字符，所以只跑 1.2ms 就通过、没能拦住。改为单遍扫描，并给该测试补上以掩码占位符收尾的形态。
- 非凭证字面量枚举改为大小写不敏感：`auth: None`、`token: Null`、`cookie: Enabled` 此前不被识别为状态字面量而判成凭证，`find_sensitive_option_paths` 据此把合法配置在边界误拒。
- 文本脱敏的界断言补齐第三条驼峰分支：`is_sensitive_option_key` 的切词含「大写串接小写词」，文本侧只实现了「非字母数字」与「小写接大写」两条，`HTTPBearer`/`HTTPSSecret` 这类首字母缩写键在配置边界判敏感、在日志里却明文输出。
- 配置校验失败不再把凭证交给解释器默认 handler：`Settings()` 在 import 期就被调用（`log_manager.get_module_logger`），早于任何入口的 `try`，pydantic 默认渲染会把 `input_value` 明文打印。现在在 `get_settings` 边界重抛已脱敏的消息，并给 `run_cli` 补上覆盖整个启动阶段的异常边界。
- 兼容渠道不再被当作官方 OpenAI 端点放行 SDK 专属资源；仅精确匹配 `https://api.openai.com/v1` 时按 SDK 契约放行。
- 收紧失败语义：凭证、SDK 或远端调用失败时显式报错，不再以空 embedding、零分 rerank 或占位回答掩盖失败。

### 移除与调整

- 移除 Dify 上游子模块；`src/etl/`、`src/retrieval/` 等移植代码继续遵守 `DIFY_LICENSE` 并保留上游版权声明。
- 清理死代码：移除 `has_system_message`、`_configured_extra_body_keys`、`SnapshotRepository.write_stats`/`load_stats`/`load_active_manifest`，以及 `recursive_text_splitter` 中未被引用的 `DEFAULT_CHILD_CHUNK_*` 常量。这些逻辑均已在调用点内联或由框架通过装饰器调用，保留会造成同一行为的两处实现。
- 移除 iFlow 渠道及其配置示例。
- 移除 qwen rerank 配置示例；该渠道不提供 rerank 能力，调用时会显式报错。
- 发布脚本新增目标环境校验，`--target` 与主机系统/架构不匹配时在清理构建产物前显式失败，不再生成错误架构的发布包。
- Qwen Base URL 改为 OpenAI 兼容接口 `https://dashscope.aliyuncs.com/compatible-mode/v1`；Volcengine Base URL 改为 Ark API `https://ark.cn-beijing.volces.com/api/v3`。配置仍为旧值时会在加载时给出显式升级警告：DashScope 原生 `api/v1` 不提供本仓库请求的 `{base_url}/chat/completions` 路径，火山旧域名 `maas-api.ml-platform-cn-beijing.volces.com` 则已停止服务。百炼的 `/responses` 只挂在业务空间专属域名下，默认共享域名不可用，文档已补充说明。
- Jina Rerank 与 SiliconFlow Rerank 一致，`return_documents` 固定为 `True` 且不允许被 `options` 覆盖。

### 测试与文档

- 新增 Provider 协议适配、SDK 能力、凭证边界、Rerank 契约与失败语义回归测试；测试总数增至 1262。新增断言经红绿验证（回退对应源码后测试变红）；其中一部分是**反向守卫**——回退源码不会变红，需要定向变异（例如「让守卫连 Ark 一起拒」）才能验证，这类断言守的是「不能过度收紧」，同样有区分度。
- 修复三处测试有效性缺陷：掩码尾部可见片段的断言用 `startswith("[REDACTED]")`，而掩码前缀在任何实现下都会被替换成 `[REDACTED]`，该断言恒真——尾部片段原样泄漏时也能通过，改为全文相等；短纯字母凭证的 7 个用例里有 4 个取值 ≥12 字符，旧规则本就能命中、对被测属性零区分度，改为真正短于 12 字符的值并断言值本身被替换；掩码线性度用例用单次采样配 100ms 阈值，实测本机 p99.9 即到 100ms、满载时更高，会间歇误报，改为放大规模到线性与二次相差三个数量级、取多轮最小值。
- 为活动快照的 embedding 兼容性检测补充回归测试：快照记录的 embedding provider 或模型名与当前运行配置不一致时必须显式失败，避免用错向量空间后静默产出错误检索结果。
- 引入 Ruff、Bandit 与 MyPy 到开发依赖，并补齐对应配置；新增 `.github/workflows/quality.yml`，在 PR 与 `main` 推送时执行格式化检查、lint、类型检查、安全扫描与完整测试，此前这些工具只在本地手动运行。
- 全仓库应用 `ruff format`（行宽 100、双引号、4 空格），此前未配置 formatter，`main.py`、`src/`、`scripts/`、`tests/` 中存在混用单引号、行尾空白和手工对齐等不一致；格式化只改表示不改语义，已用 AST 比对确认语法树等价，`AGENTS.md` 的质量检查与风格段落同步更新。
- 更新 `README.md`、`AGENTS.md` 与 `docs/`，同步协议选择、`options` 边界、原生资源入口和 Vertex ADC 配置口径。
- 核对各渠道官方现状并刷新配置示例：Google Embedding 改用 `gemini-embedding-2`（`text-embedding-004` 已于 2026-01-14 退役），OpenAI 改用 `gpt-5.6-*`（`gpt-3.5-turbo` 于 2026-10-23 退役），Anthropic 改用 `claude-sonnet-4-6`（`claude-3-5-sonnet-20240620` 已于 2025-10-28 退役），DeepSeek 改用 `deepseek-v4-pro`/`deepseek-v4-flash`（`deepseek-chat` 已于 2026-07-24 退役），火山改用 `doubao-seed-2-0-lite-260428` 与 `doubao-embedding-text-240715`（`doubao-pro-32k`、`bge-large-zh` 已不在方舟模型列表），Qwen 改用 `qwen3.8-max`/`qwen3.7-plus`，SiliconFlow 改用 `Qwen/Qwen3.5-27B`/`deepseek-ai/DeepSeek-V3.2`（复核发现 `Qwen/Qwen3-8B` 已不在该平台在售列表，其 Qwen 对话模型现从 Qwen3.5-27B 起步），Grok 改用 `grok-4.6`（`llama3-70b-8192` 实为 Groq 的 ID，不属于 xAI），Ollama 改用 `llama3.1`/`gemma3`。示例配置的键名同步改为与模型一致，模型内置默认值一并更新；内置 embedding 与 rerank 兜底值此前含无效 ID（Google 的裸名 `embedding-001`、SiliconFlow 不存在的 `alibaba/` 命名空间与已下线的 `bge-reranker-large`、Ollama 的 `llama3`），现分别改为 `gemini-embedding-2`、`BAAI/bge-large-zh-v1.5`、`BAAI/bge-reranker-v2-m3` 与 `llama3.1`。
- 需要重建知识库快照：`gemini-embedding-2` 与 `text-embedding-004` 的向量空间不兼容，旧 FAISS 索引不能复用。火山的 `doubao-embedding-text-240715` 已于 2025-12-26 停止新购（EOM），且已不在方舟「向量化能力」模型列表中；官方下线公告给出的迁移目标是 `doubao-embedding-vision-251215`，该模型同时接受纯文本输入，但它按 `/api/v3/embeddings/multimodal` 提供，与本项目使用的文本 `embeddings.create` 路径不同，迁移前需实测。embedding 模型只有 EOM 阶段、不涉及 EOS，存量接入点不受影响，因此示例配置暂未改动该值。
- 修复 Anthropic 采样字段弃用识别：家族名不再限定 `opus`/`sonnet`/`haiku`，覆盖 5 代新增的 `claude-fable-5`、`claude-mythos-5` 等命名；此前这些模型会被透传 `temperature`/`top_p`/`top_k`，而 Python SDK v1.0+ 已移除这些参数，请求会直接失败。

### 审查修复

- **补回 `GoogleProvider.embed_query`／`aembed_query` 丢失的空结果兜底**：基类 `TextEmbeddingModel.embed_query` 先判 `if not vectors` 再取 `[0]`，空结果得到带指引的 `ValueError("Embedding provider 返回空结果。")`；`GoogleProvider` 为设 `task_type` 覆写了这两个方法，却把那条兜底一并丢掉，直接下标取 `[0]`——provider 返回空列表时抛裸 `IndexError: list index out of range`。空返回不是假想输入：`embed_documents` 的条数校验按 `len(texts)` 做，传空 `texts` 时 `len(values) == len(texts) == 0` 成立、方法正常返回 `[]`；网关截断响应或返回空 `embeddings` 字段也走到同一形态。`IndexError` 不是 `ValueError`／`RuntimeError`，会穿透调用方按这两类做的错误分类与日志脱敏，消息也与「provider 返回了什么」毫无关系。现按基类逐字对齐补回，`task_type` 覆写的理由保持不变。
- **补齐父块索引写入侧的顶层形态守卫**：`register_parent_documents` 直接写 `for ... in parent_documents.items()`，非映射传进来时抛裸 `AttributeError: 'int' object has no attribute 'items'`；而载入侧的 `_normalize_documents` 对同一形态给的是带指引的 `ValueError: 知识快照不自洽：父块索引的顶层不是映射（int）…`。两者都是 `VectorStoreBase` 的公开契约方法，而 `_normalize_documents` 的注释本就写明「直接调它们的嵌入方不经过那道校验」——同一取舍只在一侧执行了。本仓自带流程到不了写入侧这条分支（`knowledge_build_service` 传的是 `dict`，且前面已用 `isinstance(..., dict)` 过滤），因此这是一致性修复而非崩溃修复。
- **补齐向量载入侧的非有限值校验**：写入侧 `EmbeddingService._as_float32_matrix` 早就拒绝 NaN／inf 向量，理由也写在注释里（FAISS 把 NaN 向量照单收下，`ntotal` 正常增加，既不报错也不跳过），但两个载入入口 `FaissStore.load`（legacy pkl）与 `load_snapshot` 共用的 `_validate_embeddings_shape` 只判 `ndim`——同一份内容**写入侧拒绝、载入侧放行**。实测载入后 `semantic_search` 对任意查询都只返回好文档、坏文档的下标干脆不出现，用户看到的是「搜到了」而不是「这份快照坏了」；而 `upsert_embeddings` 是 `VectorStoreBase` 的公开契约方法，直接调它的嵌入方绕开 `EmbeddingService`，坏向量还会被 `save_snapshot` 落盘、由 `validate_snapshot_dir` 放行成活动快照。判据落在 `astype(np.float32)` **之后**：`1e300` 在 float64 里是有限的，cast 成 float32 才溢出成 `inf`——只对原数组调 `isfinite` 会让载入侧在这一点上再次与写入侧分叉。cast 只是校验用副本，失败时 store 仍保持原状。
- **补齐配置编辑器模型菜单的容器形态校验**：`edit_model_params` 原先直接写 `list(chat_config["llm_configurations"].keys())`，取值不是映射时抛裸 `AttributeError: 'int' object has no attribute 'keys'`——不是 `ValueError`，会以裸 traceback 终止会话（主循环只捕获 `KeyboardInterrupt`／`EOFError`）。现经 `_configuration_keys` 收敛，判据与 `display_utils._require_model_detail` 一致。**只判容器、不判取值**：`display_chat_config` 在上游读值时会直接拒绝形态坏掉的配置项，菜单是用户唯一还能把激活项切走的入口，按取值判会让配置写坏后彻底没有修复路径——这条取舍由 `test_edit_model_params_lists_unusable_details_without_reading_them` 锁住。
- **补齐配置映射「取值」与「容器」两处形态校验**（同一缺陷类，`llm_configurations`／`embedding_configurations`／`rerank_configurations` 三个角色同源）：
  - **取值不可用**：`key in mapping` 只证明**键**在，不证明**值**可用。`{"k": None}` 的键确实存在，于是 `ModelProviderFactory._get_model_detail` 不会走「未找到key」那条分支，而是把 `None` 原样返回给调用方；三个入口紧接着都读 `config.provider`，在那里抛裸 `AttributeError: 'NoneType' object has no attribute 'provider'`——它不是 `ValueError`，穿透 `Chatbot.reload_llm` 的 `except ValueError` 分类。取值来源是 `SessionConfig.llm_configurations` 等可直写映射（`_VALIDATORS` 明确不登记这几个字段）。现新增 `require_detail_attributes` 统一收敛，三处调用点（工厂、`RetrievalService._get_rerank_provider`、`display_utils._require_model_detail`）共用。**判据用属性存在性而不是 `isinstance(detail, ModelDetail)`**：使用方只要求取值提供这几个属性，按具体类型收窄会把仓库里用 `SimpleNamespace` 构造的等价配置一并拒掉（实测一次按类型收窄让 17 个既有用例变红）。
  - **容器不可迭代**：`_get_model_detail` 直接写 `provider_key not in configurations`。取值不可迭代时抛裸 `TypeError: argument of type 'int' is not iterable`；是字符串时走**子串**判定（`"k" in "k"` 为真），随后 `"k"["k"]` 抛 `TypeError: string indices must be integers`。两者都不是 `ValueError`。取值经 `Chatbot.reload_llm` 从 `session_config.llm_configurations` 原样传到这里。
- **修复 `RetrievalService._get_rerank_provider` 对 rerank 缓存键两个分量的形态假设**：
  - `provider`／`model_name` 此前只经 `getattr(..., "")` 取值，有默认值但**没有类型判据**。`ModelDetail` 是 pydantic 模型、**构造**时会拒绝非字符串，但普通属性没有赋值守卫（`validate_assignment` 未开）——实测 `detail.provider = ["p"]` 与 `detail.model_name = ["m"]` 都直接成功。两者都进 `cache_key` 元组，在 `self._rerank_providers.get(...)` 处抛裸 `TypeError: unhashable type: 'list'`。
  - 属性被 `del` 掉时（实测 `del detail.provider` 成功）`getattr` 兜出的空串会把「属性没了」伪装成「provider 名为空」，一路带到工厂再抛 `AttributeError`。
  - 新增 `_is_hashable`：判据用**可哈希**而不是 `isinstance(..., str)`，非字符串但可哈希的取值（整数、元组）本来就能正常工作，按类型收窄会误伤；位数超限的整数是可哈希的，不必在此拦截。
- **修复注释里的 `type:` 被 mypy 当成类型注释**：`#     type: 'list'``；` 这类行在 Python 解析器看来是普通注释（`ast.parse`／`ruff`／`compileall` 全部通过、测试全绿），但 mypy 会把紧跟缩进后的 `type:` 识别为**类型注释**（PEP 484 风格）并去解析其后的内容，于是报 `Invalid syntax` 且**阻断整个项目的类型检查**（`errors prevented further checking`）。本次是写注释时踩到的，改动只是把该行拆成两句。记录在此是因为该类行的失败模式很隐蔽：语法错误只在 mypy 出现，其余工具全绿。

- 恢复 `src/etl/splitters/recursive_text_splitter.py` 的 Dify 移植声明头。该文件在引入 tiktoken 与层级分片的重写中被误删头部，此后 `sync dify rag hierarchical retrieval` 又向这个已无头的文件移植了父子分片逻辑，导致文件含 Dify 衍生代码却没有版权与许可证声明，与 `AGENTS.md` 的合规要求及 Apache 2.0 第 4(b) 条不符。
- Responses 错误出口补齐脱敏：`_raise_for_response_error` 此前把服务端消息直接拼进异常，而同一适配器的 Chat Completions 出口已经过 `redact_sensitive_text`。服务端错误常回显请求头或 URL，异常文本会进日志与终端，是凭证最容易泄漏的出口，现两个出口行为一致。
- 修复召回测试 Excel 记录的落盘位置：`ExcelLogger` 硬编码相对路径 `data/logs`，而文本日志使用经 `ROOT_DIR` 解析的 `settings.log_path`。相对路径基于 CWD，从仓库外运行程序时两者会分叉，Excel 记录落到当前工作目录。现默认复用 `settings.log_path`。
- `capability_report` 与 `protocol_status` 对齐角色语义：前者在 embedding/rerank 角色下仍报告 `server_verified_protocols` 并接纳运行期登记，而后者对同一输入显式报错。协议只适用于 LLM 角色，两个诊断入口现给出相同结论。
- 补充重排乱序路径与重排契约的回归测试：既有替身恒返回 `[0]`（恒等置换），`_rerank_if_needed` 的重排与按分数降序排序从未被执行，`_validate_rerank_output` 的布尔混入与越界 index 分支也无覆盖。
- 修复无断言测试：`test_rerank_accepts_top_n_larger_than_document_count` 只调用 `_validate_inputs` 而无断言，即使行为变更也不会失败；现补断言并新增 `top_n` 非法值（含 `bool` 混入）的参数化用例。
- 文档校正：`REFACTORING_PLAN.md` 标注为 v1.2.0 历史归档并修正三处与现状不符的表述（`PipelineManager` 实为 `Pipeline`、依赖应声明在 `pyproject.toml`、当时配置格式为 `config.ini`）；`README.md` 移除已退役的模型举例、修正 `/config` 可调项描述、补全 `src/` 目录树、把许可说明指向实际存在的许可证文件；`getting-started.md` 移除不存在的 `exit` 退出方式；`developer-guide.md` 补全目录树。
- 修复 Anthropic 响应解析在字段为 `None` 时崩溃：`field()` 只在键不存在时返回默认值，而 SDK 的 `thinking`/`redacted_thinking` 块会给出 `None`，此时 `"".join(...)` 抛 `TypeError: expected str instance, NoneType found`。非流式的正文与推理拼接、流式的 `text_delta`/`thinking_delta`/`input_json_delta` 三处一并加兜底。同类拼接在 Google 与 Volcengine 已有 `isinstance` 守卫，仅 Anthropic 缺。
- Google `tool_choice` 不再静默接受未知模式：`FunctionCallingConfigMode` 是大小写不敏感枚举，未命中时会合成一个同名成员、只发 `UserWarning`，随后被静默发往服务端；OpenAI 风格的 `{"type": "tool"}` 正会落到这里。现按已知模式显式映射，未知值报 `ValueError`，并给出可选值。`ToolConfig` 的未知键也从裸 pydantic `ValidationError` 归一为本项目的 `ValueError`。
- Responses `json_schema` 缺 `schema` 时显式报错：此前会回退成 `schema` 本身，伪造出 `{"type": "json_schema"}` 的 schema 发给服务端，调用方以为拿到了结构化输出约束而实际没有；同时 `strict` 不再被硬编码的 `True` 覆盖调用方取值。
- 补充流式重试助手的回归测试：`retry_sync_stream`/`retry_async_stream` 的建立阶段重试此前零覆盖，现覆盖「建立失败重试」「首事件后不再重试（避免重复输出）」「空流不算失败」「校验器只作用于首事件」「消费方提前退出时关闭迭代器」以及异步等价场景。
- 修复 Ark Chat Completions 工具调用缺 `type` 键：`VolcengineProvider._extract_result` 的 Chat Completions 分支只产出 `{"id", "name", "arguments"}`，而同一函数的 Responses 分支与 `openai_compatible.py` 的 `_extract_tool_calls` 都产出契约声明的扁平 `{"id", "type", "name", "arguments"}`（`model_provider.py:343` 与 `docs/user_guide/llm-providers.md:108` 双重写明）。调用方按契约回填 `role=tool` 历史时会配不上；现补 `type`（缺省 `function`）并新增两条钉住该键与逐键对齐 OpenAI 兼容侧的回归测试。
- 数值配置补齐加载期校验：`log_retention_days`、`kb_chunk_size`、`kb_chunk_overlap`、`kb_child_chunk_size`、`kb_child_chunk_overlap`、`kb_embedding_batch_size`、`chat_vector_weight`、`chat_keyword_weight` 此前接受任意数值，`kb_chunk_size=0` 会一路通过配置校验、直到 langchain 在分片阶段才抛 `chunk_size must be > 0`，负权重会反向加成分数；而 UI 层（`src/ui/config_menu.py`）对权重已有 `0..1` 校验，TOML/env 路径没有，形成不对称。现按字段加 validator，并用 `model_validator` 补上「overlap 必须严格小于 size」这条无法用单字段表达、但会让切分器无法推进的跨字段约束。

- 清理死代码与重复实现（零功能影响，单独一次提交以便审阅）：删除 `config.py` 中零调用的向后兼容层 `get_backward_compatible_configs()`（85 行）及其失效注释；删除 `LargeLanguageModel.supports()`（零调用，但**保留**它读取的 `capabilities`——那是 `factory.capabilities()` 经 `getattr` 读取的活数据）；删除 `volcengine._ARK_CLASSIFICATION_KEYS`（纯冗余，`classify()` 的三个同名形参无法落进 `**kwargs`，真正的校验由内联集合完成且完整）；删除 `faiss_store.FaissStore` 中从未获取的 `asyncio.Lock` 及其函数内局部 import；简化 `anthropic.py` 中条件永不产生可区分行为的恒等 `if/else`。
- 消除 `openai_compatible.py` 与 `model_provider.field()` 的重复实现：`_field` 与 `field` 逐字节相同却各存一份，且被 143 处调用（`self._field` 28 + `cls._field` 115）。现删除重复定义、统一复用 `field`，并按字母序补入 import。
- `volcengine` 的 4 个 helper（`_merge_options`/`_validate_model_options`/`_validated_extra_body`/`_merge_extra_body`）此前是 `openai_compatible` 的逐字节副本，现改为委托到后者，只保留一份实现，消除漂移风险。
- `jina.py` 与 `siliconflow_rerank.py` 是近似副本（行级相似度约 0.78，7 个方法同名，`_get_headers` 逐字节相同），但存在 4 处真实语义差异（`_OPTION_KEYS`、`_base_url` 来源、构造期 base_url 校验、`_parse_response` 的 `results` 校验时机）。**有意保持独立实现**，改在两侧 docstring 中写明差异清单与「改动必须同步」的约束，而不再引入一层抽象。
- 删除孤儿文件 `logo_ASCII.py`：全仓零引用、不在任何构建/CI 配置中，其 `create_gradient` 与 `main.py` 的同名函数逐字节重复，且未格式化（一旦纳入 `quality.yml` 扫描范围会直接失败）。
- 移除 4 个从未被 import 的依赖中的 2 个：`langchain` 与 `langchain-community`。保留 `langchain-core`（`langchain-text-splitters` 的必需依赖）与 `requests`（`google-genai`、`tiktoken` 的必需依赖）——它们的「未使用」是传递依赖的正常形态，删除会破坏真实依赖链。
- Google `tool_choice` 的异常文本在源头脱敏：裸 pydantic `ValidationError` 会把违规取值原文写进 `input_value=...`，而 `types.ToolConfig` 允许的键集里就有 `api_key` 这类凭证形态的键。两个消毒器（`redact_sensitive_text`、`safe_exception_text`）虽然都能拦住，但在拼消息前收口才是纵深防御。
- `release.yml` 的手动发布路径补交叉校验：`workflow_dispatch` 的 `tag`/`version` 是手填的，此前直接透传，会发出一个版本号与 `pyproject.toml`/`CHANGELOG.md` 不符的 release。现校验 `version` 与 `pyproject.toml` 一致、`tag` 等于 `v{version}`、且 CHANGELOG 已有该版本条目。

- 检索规模补齐上界：`chat_top_k` 此前只校验下界，而 `effective_top_k = chat_top_k * retrieval_candidate_multiplier` 会直通 `faiss_index.search()`，FAISS 既不报错也不截断。实测 `chat_top_k=10**9` 被接受，单次查询真实分配约 3.6 GB 数组、进程 RSS 涨 10.3 GB 并挂起 8.4 秒。现限制 `chat_top_k ≤ 10000`、`retrieval_candidate_multiplier ≤ 100`。
- 补齐父子分片的层级约束：`kb_child_chunk_size >= kb_chunk_size` 此前被静默接受，实测 `parent=child=300` 时不同父块数等于分块数（15/15），即每个父块只产出一个子块，「先粗后细」的父子检索退化为单层。现要求子分片严格小于父分片——与既有的 `overlap < size` 同属无法用单字段 validator 表达的约束。
- 混合检索权重不得同时为 0：`retrieval_service._weight_tuple()` 在 `total <= 0` 时返回 `(0.0, 0.0)`，所有候选得分为 0，再叠加 `weighted` 策略启用阈值过滤（默认 0.4）会导致**检索结果被全部丢弃且无任何报错**。单个权重为 0 仍合法（纯向量 / 纯关键词检索）。
- 修复布尔值绕过数值校验的真因：`bool` 是 `int` 的子类，而 pydantic 的 lax 模式会在 `field_validator` **之前**把 `True` 强转成 `1`，因此原有的 6 处 `isinstance(value, bool)` guard 全部是**不可达的死代码**——实测 `Settings(chat_top_k=True)`、`log_retention_days=True`、`chat_vector_weight=True` 等 10 个字段均被静默接受为 `1`/`1.0`。现给这些 validator 加 `mode="before"` 让 guard 真正生效，并新增 `_coerce_number()` 统一处理 `mode="before"` 下收到的原始输入（字符串数值来自 `.env`/TOML，必须继续支持）。
- `chat_temperature` 与 `retrieval_candidate_multiplier` 的 validator 补上同样的布尔短路（两者此前也用 `float(v)`/`int(v)` 静默接受 `True`）。
- 修复 `.env.example` 的默认渠道覆盖：`DEFAULT_LLM_PROVIDER="openai"` 是生效的环境变量，而环境变量优先于 `config.toml`，因此按示例原样复制后 `default_llm_provider` 被改成 `openai`、覆盖了 TOML 里的 `google`，导致 14 条模型配置因密钥为空串在运行期失败（而 `default_embedding_provider=local-hash`、`default_vector_store=faiss` 都选了零凭证通道，证明零凭证默认可行）。现改为注释并说明原因。
- 修复三例「假绿」测试：`test_numeric_settings_reject_illegal_values` 原先用 `match=field_name` 断言，而跨字段 `model_validator` 的报错文本里也含 `kb_chunk_size` 等字段名（「kb_chunk_overlap 必须小于 kb_chunk_size。」），删除被保护的 `validate_positive_sizes` 后这 3 例仍然全绿。现改为断言各 validator **独有的消息片段**，变异验证下失败数从 2 提升到 5。
- 补强 Ark `tool_calls` 的 `type` 回归测试：夹具原先写 `type="function"`，与硬编码值相同，因此把透传改成硬编码 `"function"` 的变异体无法被捕获（零区分度）。现夹具改用非 `"function"` 的取值，并新增一条覆盖「缺 `type` 时两侧默认值对齐」的交叉断言——此前断言只覆盖显式传值路径，两条路径的 default 一旦分叉测试仍全绿。
- 新增 12 例布尔短路测试，覆盖 `log_retention_days`、`kb_chunk_size`、`kb_chunk_overlap`、`chat_top_k`、`retrieval_candidate_multiplier`、`chat_vector_weight`、`chat_score_threshold`、`chat_temperature` 等字段拒绝 `True`。

- 修复分片在「分隔符耗尽」时完全失效：LangChain 的 `RecursiveCharacterTextSplitter._split_text` 走到 `if not new_separators: final_chunks.append(s)` 是原样追加、不再按 `chunk_size` 切，而默认配置 `kb_splitter_separators = ["###"]` 只有一项匹配项、`new_separators` 直接为空，于是**任何不含 `###` 的文档整篇变成一个块**。实测 `chunk_size=1500` 时 3600 token 的纯列表文本产出单个 3599 token 块；对真实知识库 39 个文件扫描，34 个产出超限块（子块上限 300，实测最坏 4595）。现让分隔符表统一以 `""` 收尾，兜底只在前面所有分隔符都不匹配时生效——含 `###` 的文档切分结果不变（实测 3 块/最大 252 字符）。
- 修复分片开头标点被直接删除：`_strip_leading_punctuation` 用 `re.sub(r"^[\s.。]+", "", text)` 删除开头连续的空白与句号。当分隔符本身就是句号时，`keep_separator=True` 会把上一句的句号留在下一个分片开头，于是那个句号被永久丢弃；整篇以句号开头的输入更会被判空并 `continue`，产出 0 个块。现改为**只**匹配空白（`re.sub(r"^\s+", ...)`），标点原样保留。**此条此前是假的**：上一版写的是「替换为等长空格」，被替换的字符集恰好等于紧随其后的 `strip()` 会删掉的字符集，替换出来的空格立刻又被删掉，与原始实现逐字符等价（实测 3 万条随机输入零差异，`"。" * 200` 依然产出 0 块，`separators=["。"]` 下依然丢 2.9% 的字符）。当时配套的测试也绕开了被测字符（`if char == "。": continue`），因此对缺陷零区分度。现断言改为「分片按序拼接等于原文」。
- 给知识快照的 pickle 加载补上来源边界：快照内的 `chunks.pkl`/`parents.pkl`/`lexical.index` 都是 pickle，反序列化即执行代码，而 `AGENTS.md` 要求「只从本项目生成的本地快照或明确受信的 legacy 文件加载」——此前代码不做任何来源校验，`load_snapshot(dir)` 接受任意目录、`load(path)` 接受任意路径、`FaissStore(file_path=...)` 还会隐式加载。现新增 `resolve_within()`/`ensure_trusted_source()`（两侧先 `resolve()` 再比归属，符号链接逃逸会被拦下；信任根为空时一律拒绝），并要求 `load_snapshot` 满足三件事：目录在信任根内、目录本身不是符号链接、`ACTIVE_SNAPSHOT` 标记指向的就是该目录。第三条是关键——仅「在 root 之下」不足以证明目录可用，未激活或半成品目录同样在 root 之下。
- 快照 manifest 补上 schema 版本准入：`load_manifest` 此前只 `str(data["schema_version"])` 解析、不比对，未知版本被静默接受并照着当前代码解释，而 v1→v2 调整过分块与 embedding 的落盘结构，这样会得到错误结果而不是报错。现版本不符即拒绝，消息给出快照版本与当前版本。版本判断放在 `load_manifest` 而非 `from_mapping`，后者保持纯解析。
- 快照完整性校验补上数据源自洽：`validate_snapshot_dir` 此前只查文件**存在性**。实测 `documents=3` 而 `embeddings=(2,4)`（`faiss_index.ntotal=2`）的 store 能正常落盘、通过校验、加载后零报错，之后 `semantic_search` 拿 `indices` 索引 `self.documents` 才越界。现在校验 chunks / embeddings / `stats.json` 三者行数一致，并把截断文件的解析失败（原先裸 `EOFError`）归类为「快照不自洽」。
- 修复 legacy 导入失败后的临时目录残留：`factory.py` 的 legacy 分支此前没有 `KnowledgeBuildService.build` 那样的 `finalized`/`finally` 结构，`create_temp_snapshot_dir` 之后任何失败都会在快照根留下 `.tmp-legacy-*` 并随时间累积。现照搬同一结构，并把 `validate_snapshot_dir(temp_dir)` 提到 `finalize_snapshot` **之前**——finalize 会更新 `ACTIVE_SNAPSHOT`，校验放到它后面就只剩「坏快照已经被激活」这一种收场方式。
- 修复 `SessionConfig` 完全无校验：`Settings` 只在启动时校验一次，而 UI 与库调用方经 `__setitem__` 写入的 `SessionConfig` 此前不校验任何字段。实测 `chat_config["top_k"] = 10**9` 能一路走到 `faiss_index.search()`（实测 `10**6` 时就已按满槽位分配并耗时 1.56 秒），同时接受 `vector_weight=-5`、`retrieval_method='totally-invalid'`、`score_threshold=None`。现为 9 个标量字段补上校验器注册表，并加结构断言防止漏登记——漏登记会静默回落为无校验，且不会让任何行为测试变红。
- 修复关键词检索的静默空结果：`keyword_search` 用 `if score <= 0: break`，既分不清「该词无判别力」与「0 分文档排在前面把正分结果截断」，又在分数整片为 0 时静默返回空列表。rank_bm25 的 idf 是 `log(N-n+0.5) - log(n+0.5)` 且只对 `idf < 0` 做 epsilon 浮动，某个词**恰好**出现在一半文档里（`n == N/2`）时 idf 恰为 0、不属于「负」因此不被浮动，该词分数就全是 0。现先看全局最高分，再按分数降序收集、0 分只跳过不终止。
- 修复空语料导致的 BM25 崩溃：`load_snapshot` 从 `lexical.index` 读回 `[]` 时 `BM25Okapi` 在 `avgdl = num_doc / corpus_size` 处抛 `ZeroDivisionError`。现空语料返回 `None` 并在 `keyword_search` 里短路，不再崩溃。
- 给嵌入向量矩阵补上有限性校验：`_as_float32_matrix` 此前只校验维度、非空与行数，NaN/inf 会被接受并写入 FAISS（`ntotal` 正常增加），直到查询时才以 3.4e38 的哨兵距离暴露出来，届时已无法定位是哪个文档坏了；`1e300` 还会先触发 `RuntimeWarning: overflow encountered in cast` 静默溢出为 inf。现对齐 `retrieval_service` 既有的 `math.isfinite` 校验。
- 移除退出钩子的无条件删除：`cleanup_temp_files()` 已在 `atexit` 注册、每次程序正常退出都对 `settings.cache_path` 执行 `shutil.rmtree`，但删除前没有任何安全判断，而 `cache_path` 是用户可配置项且校验只做相对路径转绝对路径。实测 `Settings(cache_path="/")`、`"/etc"`、`"/usr"`、`"~"`、`"/var"`、`"/System"`、`"../.."` 全部被接受，随后整个目录被删掉——不可逆的数据丢失，触发条件只是「配置文件写错一个值」。现拒绝文件系统根目录、家目录本身与常见系统目录，判定只看「是否危险」而不看目录名（用户可能有意把缓存放到 `data/cache`）。该缺陷不在任何一轮审查报告里，是靠盘点「从未被测试引用的生产模块」找到的。
- 修复配置数值解析的宽松放行：`_coerce_number` 的 `int()`/`float()` 会接受 `"1_0"`、`"+7"`、`"0x10"`（`int("1_0") == 10`、`float("1_0") == 10.0`），把写错的配置静默读成另一个数。现先做十进制字面量形状校验。同时修复 `chat_score_threshold=10**400` 抛裸 `OverflowError`（它不是 `ValidationError`，会穿透 `get_settings` 的 `except` 变成未脱敏 traceback），把范围判断移到转 float 之前。
- 给无上界的数值字段补上界：`log_retention_days`、`kb_chunk_size`、`kb_child_chunk_size`、`kb_embedding_batch_size` 此前只有下界，实测 `10**400` 全部被接受，直到参与运算（日期减法、内存申请）才炸。
- 修复列表类 env 字段的未处理异常：`KB_SPLITTER_SEPARATORS='###'`（非 JSON 形态）抛的是 pydantic-settings 的 `SettingsError`，它继承 `ValueError` 但**不是** `ValidationError`，原 handler 接不住，用户看到的是解释器默认 handler 打出的多屏 traceback。现转成一行脱敏消息。
- 修复发布包 ZIP 分支对悬空符号链接的裸崩溃：`ZipFile.write` 会解引用符号链接去取元数据，目标缺失时抛的是不带任何线索的 `FileNotFoundError`，而 tar.gz 分支对同一目录完全正常——两分支行为不一致，且在 Windows（唯一的 zip 目标）上表现为难以归因的构建失败。现归档前显式检查并列出全部悬空路径。经本地 `--target macos-x64 --validate` 实测，真实产物内符号链接 0 个，该检查在当前构建下属纯防御性。
- 修复 `_initialize_vector_store` 的虚假成功提示：没有快照时 store 为空（`documents=0`、`faiss_index=None`），却仍打印「知识快照加载成功。」，让用户先看到成功再看到永远是「无相关文档」，把构建缺失误诊成检索质量差。现按实际分块数区分提示。
- UI 的 `top_k` 与 `retrieval_candidate_multiplier` 输入补上界：两处 validator 此前只校验下界，接受任意大的值后再交给 `SessionConfig`。
- 收紧 pytest 的警告过滤：`pytest.ini` 原先用全局 `ignore::UserWarning` / `ignore::DeprecationWarning`，把四个快速迭代的 SDK 升级警告和本项目自身的警告一起静音（实测同一探针用例在全局忽略下静默通过、在 `-W error::UserWarning` 下失败）。现改为按第三方 module 前缀限定，并新增断言守住本仓库配置加载路径不产生这两类警告。
- 补齐此前完全空缺的测试覆盖：legacy 导入路径（含失败清理）、`archive_bundle`（此前零测试）、`cleanup_temp_files`（atexit 删除路径，失败后果不可逆）、以及 pickle 信任边界的 6 类拒绝用例与合法路径成功用例。测试总数增至 1262。

## [1.3.0] - 2026-03-20

### 运行与配置

- 将主配置迁移到 `config.toml`，并明确 `config.toml` 为本地文件，不再纳入版本控制。
- 新增 `local-hash` 本地嵌入提供方，默认知识库构建链路不再依赖外部 Embedding API。
- 新增 `iflow-qwen3-max` 的 OpenAI 兼容配置示例，并补齐 `config.toml.example` 与 `.env` 的文档口径。

### 稳定性与检索

- 修复聊天会话和召回测试中的异步输入问题，消除 `asyncio.run()` 与事件循环冲突。
- 收紧 Provider 失败契约：当凭证、SDK 或远端调用失败时，统一改为显式报错并保留日志，不再返回占位回答、空 embedding 或零分 rerank 结果。
- 修复聊天 `/config` 切换 LLM 时的状态回滚问题；当新模型加载失败时，活动模型配置会回退到旧值，避免界面状态与实际模型实例不一致。
- 修正 `grok` 与 `volcengine` Provider 的真实导入链路，并新增 Provider 导入烟测，避免“测试通过但模块无法导入”的假收敛。
- 优化知识库重建流程，向量化时不再加载旧索引追加写入。
- 增强本地检索质量：索引文本引入来源文件标题上下文，聊天检索使用意图文本而不是原始提示约束语。
- 完成父子分段检索、RRF 融合、sidecar 父文档存储等 RAG 核心能力的适配与校验。
- 将知识库持久化升级为快照目录，新增 `ACTIVE_SNAPSHOT` 指针、快照清单和按快照加载的运行时流程。
- 引入 `RunConfig`、`SessionConfig`、`KnowledgeBuildService`、`RetrievalService`、`ChatService` 等分层运行时服务。
- 修正真实交互场景中的展示与输入边界：来源路径改为相对路径，聊天空输入不再触发检索和模型调用。

### 测试与文档

- 增加多组回归测试，覆盖本地嵌入、异步交互、索引文本构建、RRF 行为、sidecar 持久化和配置默认值。
- 更新 `README.md`、`docs/` 和 `AGENTS.md`，同步当前启动方式、配置结构和运行口径。

## [1.2.0] - 2025-07-03

### 架构重构

- 完成第二阶段重构：解耦向量存储。
  - 定义了 `VectorStoreBase` 抽象基类，统一了向量存储接口。
  - 实现了 `FaissStore`，将 FAISS 逻辑封装其中。
  - 创建了 `VectorStoreFactory`，实现了向量存储的动态加载。
- 完成第三阶段重构：构建文档处理流水线。
  - 创建了 `etl` 模块，并为 `extractors`、`cleaners`、`splitters` 定义了抽象基类。
  - 实现了针对 Markdown 的抽取器、基础文本清洗器和递归文本分割器。
  - 创建了 `PipelineManager`，实现了数据处理的动态组合。
- 完成第四阶段重构：提升健壮性与开发者体验。
  - 为核心模块如模型提供商工厂、向量存储工厂、ETL 流水线编写了全面的单元测试。
  - 优化了日志记录，并完善了项目文档。

### 许可证合规性

- 检查了 Dify 的许可证，并为项目添加了 `DIFY_LICENSE` 文件。
- 为项目中非 Dify 移植代码的部分添加了 `LICENSE` 文件。
- 在所有移植自 Dify 的代码文件顶部添加了许可证声明。
- 在 `README.md` 中明确列出了所有移植自 Dify 的代码文件。

## [1.1.0] - 2025-07-02

### 架构重构

- 明确项目目标为 Dify 核心逻辑的本地验证器，通过轻量级 Python 实现对齐并验证 Dify 的核心工作流。
- 启动向配置、抽象基类和工厂模式的系统性重构。
- 完成第一阶段重构：模型管理系统。
  - 废除了在代码中硬编码模型实例的方式。
  - 建立了 `LargeLanguageModel`、`TextEmbeddingModel`、`RerankModel` 等模型抽象基类。
  - 实现了 `ModelProviderFactory`，用于根据配置动态加载和实例化不同的模型提供商。

### 新功能与优化

- 优化启动界面。
  - 新增由 `pyfiglet` 生成的 ASCII Art 启动横幅。
  - 为标题实现从左到右的蓝红颜色渐变效果。
  - 添加了包含版本、描述、作者和 GitHub 链接的欢迎面板，并确保其宽度与标题对齐。
- 重构配置系统。
  - 从 `.env` 和 `python-dotenv` 迁移到 `config.ini` 和内置的 `configparser`，以解决复杂 JSON 配置解析问题。
  - 实现环境变量优先的配置加载策略，当环境变量存在时会覆盖 `config.ini` 中的设置。
  - 更新 Google LLM 的默认模型为 `gemini-2.5-flash` 和 `gemini-2.5-pro`。
- 提升代码质量。
  - 解耦了 Rerank 提供商，通过依赖注入将 `top_n` 作为参数传递给 `rerank` 方法，移除了对全局配置的直接依赖。
