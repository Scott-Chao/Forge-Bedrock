"""Computation graph visualization for the autograd engine."""

from __future__ import annotations

from .value import Value


def trace(root: Value) -> tuple[list[Value], list[tuple[Value, Value]]]:
    """Walk the DAG and collect topologically sorted nodes and (parent, child) edges."""
    nodes: list[Value] = []
    edges: list[tuple[Value, Value]] = []
    visited: set[int] = set()

    def dfs(v: Value) -> None:
        if id(v) in visited:
            return
        visited.add(id(v))
        for parent in v._children:
            edges.append((parent, v))
            dfs(parent)
        nodes.append(v)

    dfs(root)
    return nodes, edges


def render_text(root: Value) -> str:
    """Render the computation graph as an indented tree using box-drawing characters."""
    lines: list[str] = []
    visited: set[int] = set()

    def _render(node: Value, prefix: str = "", is_last: bool = True) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))

        label = f"[{node._op}]" if node._op else "[ ]"
        line = f"{prefix}{'└── ' if is_last else '├── '}{label} data={node.data} grad={node.grad}"
        lines.append(line)

        children = sorted(node._children, key=id)
        if not children:
            return

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            _render(child, child_prefix, i == len(children) - 1)

    _render(root, "", True)
    return "\n".join(lines)


def draw_dot(
    root: Value,
    filename: str = "computation_graph",
    format: str = "svg",
    direction: str = "TB",
) -> None:
    """Render the computation graph using Graphviz."""
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            "graphviz is required for draw_dot(). Install it with: pip install graphviz"
        )

    nodes, edges = trace(root)
    dot = Digraph(filename=filename, format=format)
    dot.attr(rankdir=direction)

    for n in nodes:
        nid = str(id(n))
        label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        op_label = n._op if n._op else "input"
        label += f"<TR><TD>{op_label}</TD></TR>"
        label += f"<TR><TD>data={n.data}</TD></TR>"
        label += f"<TR><TD>grad={n.grad}</TD></TR>"
        label += "</TABLE>>"
        style = "filled" if not n._op else ""
        fillcolor = "#E8F5E9" if not n._op else "#E3F2FD"
        dot.node(nid, label=label, shape="plain", style=style, fillcolor=fillcolor)

    for parent, child in edges:
        dot.edge(str(id(parent)), str(id(child)))

    dot.render(cleanup=True)
