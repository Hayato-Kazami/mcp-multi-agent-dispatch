"""命令行入口。

python -m dispatch_agent.cli run "查一下 W1 排队的任务" --user U001
python -m dispatch_agent.cli run "把任务 T2002 派给机器人" --user U001 --manual
python -m dispatch_agent.cli tools
"""

from __future__ import annotations

import argparse
import asyncio
import json

from langgraph.types import Command

from .agents.base import DirectGateway, McpGateway
from .graph import build_graph, initial_state, summarize
from .store import Store
from .tools import STORE, mcp


async def _make_gateway(transport: str):
    if transport == "mcp":
        gateway = McpGateway(mcp)
        await gateway.connect()
        return gateway
    return DirectGateway(STORE if transport == "direct" else Store())


async def _run(args: argparse.Namespace) -> int:
    gateway = await _make_gateway(args.transport)
    graph = build_graph(gateway, args.user)
    config = {"configurable": {"thread_id": args.thread}}
    state = initial_state(args.request, args.user, args.warehouse, auto_approve=not args.manual)

    result = await graph.ainvoke(state, config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n[待审批]")
        for row in payload.get("actions", []):
            print(f"  - {row['summary']}")
        answer = input("批准执行？[y/N] ").strip().lower()
        result = await graph.ainvoke(Command(resume={"approved": answer == "y"}), config)

    print(f"\n用户：{args.user}   仓库：{args.warehouse}   传输：{args.transport}")
    print(f"指令：{args.request}")
    print(f"回答：{result.get('answer')}")
    print(f"步数：{result.get('steps')}")
    for row in result.get("results", []):
        print(f"  - {row['action']['tool']:<12} ok={row['result'].get('ok')} code={row['result'].get('code')}")

    if args.show_trace:
        print("\ntrace:")
        print(json.dumps(result.get("trace", []), ensure_ascii=False, indent=2))
    if isinstance(gateway, McpGateway):
        await gateway.close()
    return 0


async def _list_tools(transport: str) -> int:
    gateway = await _make_gateway(transport)
    if not isinstance(gateway, McpGateway):
        print("tools 列表只在 mcp 传输下可用")
        return 1
    for tool in await gateway.list_tools():
        print(f"- {tool['name']}: {tool['description']}")
    await gateway.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dispatch-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="执行一条自然语言调度指令")
    run.add_argument("request")
    run.add_argument("--user", default="U001")
    run.add_argument("--warehouse", default="W1")
    run.add_argument("--transport", choices=["mcp", "direct"], default="mcp")
    run.add_argument("--thread", default="cli-1")
    run.add_argument("--manual", action="store_true", help="写操作走人工审批（interrupt）")
    run.add_argument("--show-trace", action="store_true")
    run.set_defaults(func=lambda args: asyncio.run(_run(args)))

    tools = sub.add_parser("tools", help="列出 MCP 工具")
    tools.add_argument("--transport", choices=["mcp", "direct"], default="mcp")
    tools.set_defaults(func=lambda args: asyncio.run(_list_tools(args.transport)))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
