"""Action：Agent 的"提议"，不是"执行"。

Agent 只负责产出 Action 与理由；执行、鉴权、幂等全部在工具层。
这个切分让人在环审批有地方可插：审批看的就是 Action 的 summary。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Action:
    tool: str
    args: dict
    rationale: str
    write: bool = False
    summary: str = ""
    fallbacks: list[dict] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "tool": self.tool,
            "args": self.args,
            "write": self.write,
            "summary": self.summary or self.rationale,
        }
