# mcp-multi-agent-dispatch

**MCP 工具层 + LangGraph 三代理编排**：调度员说一句话，系统查队列、选机器人、算路径、派单、上报故障，写操作走人工审批，重复提交走幂等。

不是"多代理聊天演示"：13 条端到端用例全部走**真 MCP 协议**（in-memory transport）和**真 LangGraph 状态机**，覆盖越权、状态冲突、信息不足、重复提交四类边界。

| 指标 | 实测 |
|---|---|
| 端到端用例通过率 | **13/13** |
| 平均步数（每条指令） | **3.69** |
| 工具调用总数（13 条用例） | **58 次**（执行 13 + 只读 45） |
| 越权拦截 | **2/2**（viewer 写操作、跨仓查询） |
| 澄清正确率 | **1/1**（缺任务号时不猜） |
| MCP 工具数 | 6（4 只读 + 2 写） |
| **提案采纳率** | **13/13 = 1.00** |
| **执行阶段重试倍数** | **1.00x**（没有一次动用 fallback） |
| **决策读放大** | **3.46 次只读调用 / 每个写提案** |
| **停滞断路器触发次数** | **0**（当前图是 DAG，本来就不会触发） |

### Agent 级指标：正例和负例必须分开算

| 代理 | 提案数 | 正例成功 | 负例正确拒绝 |
|---|---|---|---|
| AssignAgent | 7（正 2 / 负 5） | 1.00 | 1.00 |
| CollabAgent | 5（正 4 / 负 1） | 1.00 | 1.00 |
| PathAgent | 1（正 1 / 负 0） | 1.00 | — |

如果把正负例混在一起算"工具成功率"，AssignAgent 会掉到 **0.17**——看上去像这个代理基本不工作。真实原因是 5 条负例（低电量、任务已被占用、越权）**本来就该被工具拒绝**，拒绝是正确行为。

**这也解释了为什么没做 token 成本与通信效率**：这个项目里 `planner` 是正则规则、三个代理是顺序函数调用，既没有 LLM 调用也没有 agent 间消息。那两项指标在这里没有定义，硬凑出来就是编的。

## 跑起来

```bash
pip install -e .

python -m dispatch_agent.cli tools                       # 列出 6 个 MCP 工具
python -m dispatch_agent.cli run "查一下 W1 排队的任务" --user U001
python -m dispatch_agent.cli run "把任务 T2002 派给机器人" --user U001
python -m dispatch_agent.cli run "R102 一级故障，马上处理" --user U002 --manual   # 走人工审批
python scripts/eval_graph.py                             # 13 条端到端评测
```

`--transport direct` 可绕过 MCP 直调工具实现，只用于排查是"协议问题"还是"逻辑问题"。

**要求 Python ≥ 3.11**。人工审批那条路径（`--manual`）用 `interrupt()` 实现，它靠 ContextVar 取 `thread_id`；
把 context 传进工作线程的 `run_coroutine_threadsafe(context=...)` 是 3.11 才有的参数，3.10 上会在审批时抛
`Called get_config outside of a runnable context`。同步节点不受影响，所以症状是"别的都能跑，只有 `--manual` 挂"——
这是解释器版本问题，不是图写错了。原因与验证过程见 [docs/design.md](docs/design.md) §3。

## 架构

```
用户指令
   │
   ▼
┌──────────────┐   规则化解析：intent + slots + 缺失项
│  understand  │   缺参数 → clarify（不猜）
└──────┬───────┘
       ▼
┌──────────────┐   AssignAgent（单任务选车）
│   propose    │   PathAgent（只读路径）
└──────┬───────┘   CollabAgent（批量/异常）
       ▼
┌──────────────┐   只有 write=True 的动作进审批
│    review    │   interrupt + checkpointer，断点可恢复
└──────┬───────┘
       ▼
┌──────────────┐   失败按 fallbacks 换候选；步数上限 8；动作指纹去重
│   execute    │
└──────┬───────┘
       ▼
    finalize  →  中文答复 + 每个工具的执行码

   工具层（MCP）：query_tasks / list_robots / get_battery / plan_path
                  assign_task / report_fault
                  鉴权 → 幂等 → 资源存在 → 业务规则 → 执行
```

## 三个关键设计

**① 授权在工具层，不在 Agent 层。**
Agent 可以"提议"任意操作，但工具自己决定执不执行。模型被诱导、Agent 写错逻辑，都越不过这条线。这也是为什么 `authorize()` 在 `_assign_task` 的第一行，而不是在 graph 里。

**② 写操作与只读操作共用一条图，只有写操作进审批。**
只读（查队列、算路径）直接执行，写（派单、报障）必须过 `review`。判断依据是 `Action.write`，不是"哪个 agent 调的"——同一个 agent 可能既查又写。
审批用 `interrupt()` + checkpointer：状态落在 MemorySaver 里，进程重启后能续跑，这是"断点恢复"和"假装断点恢复"的区别。

**③ 防死循环靠三道，但当前这张图其实用不上。**

| 防护 | 防什么 | 什么时候才真的生效 |
|---|---|---|
| 步数上限 `MAX_STEPS = 8` | 停不下来 | 加了回边之后 |
| 动作指纹去重（`tool + sorted(args)`） | 两个动作来回跳 | 同上 |
| **停滞断路器**（连续 2 次工具失败即停手） | 每轮参数都不同但一直在错 | 一个请求产生多个写动作时 |

**说清楚一件事**：当前图是 DAG（understand → propose → review → execute → finalize），没有回边，一个请求最多产生一个写动作，所以这三道在 13 条用例下一次都没触发——`stall_events = 0` 是正确结果，不是没实现。它们是给"加反思回边"预留的保险丝。

诚实一点讲这件事比硬说"我做了防死循环"更值：**能用代码证明它有效的前提，是先说清它什么时候才需要生效。**

## 业务规则（都在 `store.py`，改这里不影响编排）

| 规则 | 值 | 拒绝码 |
|---|---|---|
| 派单最低电量 | 20% | `BATTERY_TOO_LOW` |
| 电池健康度下限 | 80% | `ROBOT_FAULT` |
| 充电提醒阈值 | 25% | （只提示，不拦截） |
| 同一巷道作业上限 | 2 台 | `LANE_CAPACITY_EXCEEDED` |
| 图搜索步数上限 | 8 | `STEP_LIMIT` |

机器人与任务数据的口径见 `data/mock_business.json`：3 个仓库、11 台机器人、7 条任务、4 个角色。

## 评测怎么做的

`scripts/eval_graph.py` 里 13 条用例，每条**重置内存库**再跑（用例间状态污染是这类测试最常见的假通过来源），断言落在三处：

1. **最终回答**（是否说清"派给了谁""路径是什么"）
2. **每个工具的返回码**（`BATTERY_TOO_LOW` / `UNAUTHORIZED` 这些不是异常，是业务语义）
3. **批内一致性**（一次派三单不能把同一台机器人派两次）

刻意设计的三类用例：

- **点名但不可用**：用户说"派给 R103"，而 R103 电量 18% → 必须拒绝，**不能悄悄换成别的机器人**。所以点名时 `fallbacks` 为空。
- **越权**：viewer 派单、dispatcher 查别的仓 → 由工具拒绝，不是 Agent 拒绝。
- **重复提交**：同一条派单连跑两次 → 第二次返回 `IDEMPOTENT_REPLAY`，不重复下发给机器人。

### 三个指标的定义与局限

| 指标 | 定义 | 它能说明什么 / 不能说明什么 |
|---|---|---|
| **提案采纳率** 1.00 | 被执行的动作数 ÷ propose 产生的动作数 | 说明"提议的东西都真的去做了"，没有生成完就丢；**不能**说明提议对不对（对不对看正/负例成功率） |
| **执行重试倍数** 1.00x | 执行阶段的工具调用数 ÷ 提案数 | 1.00 表示主方案一次成功、没动用 fallback；>1 说明候选选择质量下降 |
| **决策读放大** 3.46x | 只读调用数 ÷ 写提案数 | 每次派单平均要查 3.5 次队列/机器人/路径。这是**下一个优化点**：机器人状态缓存能把这块压到 1–2 次 |

⚠️ 这几个数都只来自 **13 条用例**，属于"能证明链路通、不能当性能基准"。`3.46` 这个数尤其不能外推——它完全取决于 `assign` 里查了几次、语料里几个仓几台车。

## 与真实系统的差异（别把它当成生产代码）

| 这里 | 生产里 |
|---|---|
| 规则化 `planner.py` 做意图识别 | LoRA 微调的 14B 意图分类（同一接口：intent + slots + 缺失项） |
| 内存态 `Store` + JSON mock 数据 | MySQL + Redis，工具层走真库 |
| 单机 in-memory MCP transport | 独立 MCP Server 进程，stdio/SSE 传输 |
| 审批用命令行 `input()` | 前端确认卡片 + 逐项批准（一次只能批一项） |
| 没有 LLM 参与决策 | 主 Agent 用云端模型规划，工具返回码驱动下一步 |

保留规则化解析是故意的：换成模型后，每个失败都会变成"模型没理解"这种不可定位的原因，评测数字就不可复现了。**先把编排和边界验证对，再换模型**。

## 目录

```
src/dispatch_agent/
  models.py      枚举与数据结构
  store.py       mock 业务库 + 幂等表 + Dijkstra 路径
  auth.py        工具级 + 仓库级双层鉴权
  tools.py       FastMCP 6 个工具（唯一能改状态的入口）
  planner.py     规则化意图解析
  agents/        AssignAgent / PathAgent / CollabAgent + 工具网关
  graph.py       LangGraph 状态机（含 interrupt 与防死循环）
  cli.py         命令行入口
data/mock_business.json   3 仓库 / 11 机器人 / 7 任务 / 4 角色
scripts/eval_graph.py     13 条端到端评测
results/eval_report.json  实测报告
```

## License

MIT —— 见 [LICENSE](LICENSE)。`data/mock_business.json` 为虚构业务数据。
