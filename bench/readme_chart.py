"""Writes the README's head-to-head chart as one theme-aware SVG.

    python3 bench/readme_chart.py docs/assets/benchmark-fiqa.svg

Values come from bench/results/2026-10-25-cloud-4core-h2h (other frameworks),
2026-10-26-quality-sweep (Gyrfalcon nDCG@10), and 2026-10-26-lite-refresh (Gyrfalcon speed and
memory)."""
import sys

SYSTEMS = ["Gyrfalcon", "LlamaIndex", "Haystack", "LangChain"]
PANELS = [
    ("nDCG@10", "higher is better", [0.266, 0.242, 0.239, 0.188], lambda v: f"{v:.3f}"),
    ("Index build", "seconds, lower is better", [2.0, 32.8, 17.7, 20.6], lambda v: f"{v:,.1f} s"),
    ("Query p50", "milliseconds, lower is better", [2.7, 1436, 2226, 2398], lambda v: f"{v:,.1f} ms" if v < 10 else f"{v:,.0f} ms"),
    ("Peak memory", "MB, lower is better", [414, 2116, 2386, 2791], lambda v: f"{v:,} MB"),
]
W, LABEL_W, GAP, TOP, ROW, BAR_H = 1000, 112, 18, 92, 40, 20
PANEL_W = (W - LABEL_W - 24 - GAP * (len(PANELS) - 1)) / len(PANELS)
BAR_MAX = PANEL_W - 74
H = TOP + ROW * len(SYSTEMS) + 54

out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
       'aria-label="Gyrfalcon against LlamaIndex, Haystack, and LangChain on FiQA: nDCG@10, index time, query latency, peak memory">',
       """<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #1f2328; }
  .muted { fill: #59636e; }
  .title { font-size: 17px; font-weight: 600; }
  .sub { font-size: 12px; }
  .label { font-size: 14px; }
  .value { font-size: 13px; }
  .strong { font-weight: 700; }
  .ours { fill: #111418; }
  .bar { fill: #c8ccd1; }
  .rule { stroke: #d1d9e0; stroke-width: 1; }
  @media (prefers-color-scheme: dark) {
    text { fill: #e6edf3; }
    .muted { fill: #9198a1; }
    .ours { fill: #f0ede7; }
    .bar { fill: #4a5059; }
    .rule { stroke: #3d444d; }
  }
</style>"""]
for i, name in enumerate(SYSTEMS):
    y = TOP + i * ROW + BAR_H / 2 + 5
    cls = "label strong" if i == 0 else "label"
    out.append(f'<text x="{LABEL_W - 12}" y="{y:.1f}" text-anchor="end" class="{cls}">{name}</text>')
for p, (title, sub, values, fmt) in enumerate(PANELS):
    x0 = LABEL_W + p * (PANEL_W + GAP)
    out.append(f'<text x="{x0:.1f}" y="40" class="title">{title}</text>')
    out.append(f'<text x="{x0:.1f}" y="60" class="sub muted">{sub}</text>')
    out.append(f'<line x1="{x0:.1f}" y1="{TOP - 10}" x2="{x0:.1f}" y2="{TOP + ROW * len(SYSTEMS) - 14}" class="rule"/>')
    top = max(values)
    for i, v in enumerate(values):
        y = TOP + i * ROW
        w = max(BAR_MAX * v / top, 2.5)
        cls = "ours" if i == 0 else "bar"
        out.append(f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{BAR_H}" rx="3" class="{cls}"/>')
        vcls = "value strong" if i == 0 else "value muted"
        out.append(f'<text x="{x0 + w + 8:.1f}" y="{y + BAR_H / 2 + 5:.1f}" class="{vcls}">{fmt(v)}</text>')
out.append(f'<text x="{LABEL_W}" y="{H - 16}" class="sub muted">FiQA-2018: 57,638 documents, 648 queries. 4 vCPU, same embedding model '
           '(potion-retrieval-32M) for every system. Linear scales. Source: bench/results/.</text>')
out.append("</svg>")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
