"""Turn a compiled LangGraph into Graphviz DOT so Streamlit can draw it."""


def to_dot(compiled_graph) -> str:
    g = compiled_graph.get_graph()
    lines = [
        "digraph G {",
        'rankdir=LR; bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fillcolor="#2a78d6", fontcolor="white", fontname="Helvetica", color="#2a78d6"];',
        'edge [color="#8a8984"];',
    ]
    for node_id in g.nodes:
        if node_id in ("__start__", "__end__"):
            label = "START" if node_id == "__start__" else "END"
            lines.append(f'"{node_id}" [label="{label}", shape=oval, fillcolor="#52514e", color="#52514e"];')
        else:
            lines.append(f'"{node_id}" [label="{node_id}"];')
    for e in g.edges:
        style = ', style="dashed"' if e.conditional else ""
        lines.append(f'"{e.source}" -> "{e.target}" [{style.lstrip(", ")}];')
    lines.append("}")
    return "\n".join(lines)
