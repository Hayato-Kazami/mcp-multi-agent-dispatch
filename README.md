# mcp-multi-agent-dispatch

**MCP 工具层 + LangGraph 三代理编排**：调度员说一句话，系统查队列、选机器人、算路径、派单、上报故障，写操作走人工审批，重复提交走幂等。

不是"多代理聊天演示"：13 条端到端用例全部走**真 MCP 协议**（in-memory transport）和**真 LangGraph 状态机**，覆盖越权、状态冲突、信息不足、重复提交四类边界。

| 指标 | 实测 |
|---|---|
| 端到端用例通过率 | **13/13** |
| 平均步数（每条指令） | **3.69** |
| 工具调用总数（13 条用例） | **58 次** |
| 越权拦截 | **2/2**（viewer 写操作、跨仓查询） |
| 澄清正确率 | **1/1**（缺任务号时不猜） |
| MCP 工具数 | 6（4 只读 + 2 写） |

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

**③ 防死循环靠两样，缺一不可。**
步数上限 8 防"停不下来"，动作指纹去重防"两个动作来回跳"。只做前者会在循环里白跑满 8 步，只做后者会在参数每轮都不同时停不下。

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
