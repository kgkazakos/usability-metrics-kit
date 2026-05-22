"""Streamlit UI for usability-metrics-kit.

Run with:  streamlit run usability_metrics_kit/app.py
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from usability_metrics_kit import llm
from usability_metrics_kit.models import Direction
from usability_metrics_kit.registry import REGISTRY, by_family, family_label

st.set_page_config(page_title="Usability Metrics Kit", page_icon="◆", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #fbfaf8; }
h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; letter-spacing: -0.02em; }
.umk-mono { font-family: 'DM Mono', monospace; }
.umk-title { font-size: 1.55rem; font-weight: 700; color: #1a1a18; margin-bottom: 0; }
.umk-sub { color: #8a877f; font-size: 0.86rem; font-family: 'DM Mono', monospace; }
.umk-card { background: #fff; border: 1px solid #ebe8e1; border-left: 3px solid #d85a30;
            border-radius: 10px; padding: 20px 22px; margin-bottom: 14px; }
.umk-score { font-family: 'DM Mono', monospace; font-size: 2.7rem; font-weight: 500;
             color: #1a1a18; line-height: 1; }
.umk-ci { font-family: 'DM Mono', monospace; color: #8a877f; font-size: 0.8rem; }
.umk-label { color: #8a877f; font-size: 0.72rem; text-transform: uppercase;
             letter-spacing: 0.08em; font-weight: 500; }
.umk-chip { display: inline-block; background: #f3efe7; color: #5f5e5a; border-radius: 5px;
            padding: 2px 9px; font-size: 0.74rem; margin-right: 5px; font-family: 'DM Mono', monospace; }
.umk-chip-good { background: #e1f5ee; color: #0f6e56; }
.umk-narr { background: #f7f4ee; border-radius: 8px; padding: 12px 15px; font-size: 0.9rem;
            color: #3a3a37; margin-top: 10px; }
.umk-note { color: #9a7b3a; font-size: 0.78rem; margin-top: 4px; }
.stButton>button { background: #1a1a18; color: #fff; border-radius: 7px; border: 0;
                   font-family: 'DM Sans'; font-weight: 500; }
.umk-steps { margin: 4px 0 2px; }
.umk-step { display: flex; align-items: center; gap: 9px; font-size: 0.82rem;
            color: #3a3a37; margin-bottom: 8px; }
.umk-step span { display: inline-flex; align-items: center; justify-content: center;
                 width: 19px; height: 19px; border-radius: 50%; background: #d85a30;
                 color: #fff; font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
                 font-family: 'DM Mono', monospace; }
.umk-fam { font-size: 0.74rem; font-weight: 700; color: #1a1a18; margin: 9px 0 3px;
           text-transform: uppercase; letter-spacing: 0.04em; }
.umk-fam small { color: #b0aca2; font-weight: 500; }
.umk-mlist { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8a877f;
             line-height: 1.5; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="umk-title">◆ Usability Metrics Kit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="umk-sub">SUS · UMUX · SUPR-Q · PSSUQ · UEQ-S · NASA-TLX · '
    'task & loyalty metrics — scored, benchmarked, narrated</div>',
    unsafe_allow_html=True,
)
st.write("")

# --- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("#### How it works")
    st.markdown(
        "<div class='umk-steps'>"
        "<div class='umk-step'><span>1</span> Choose a metric family</div>"
        "<div class='umk-step'><span>2</span> Pick the metric</div>"
        "<div class='umk-step'><span>3</span> Paste or upload your data</div>"
        "<div class='umk-step'><span>4</span> Score &amp; read the interval</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("#### Configure")
    grouped = by_family()
    family_names = {f: family_label(f) for f in grouped}
    fam_choice = st.selectbox(
        "Metric family", options=list(grouped), format_func=lambda f: family_names[f]
    )
    specs = grouped[fam_choice]
    spec = st.selectbox(
        "Metric", options=specs, format_func=lambda s: f"{s.name} ({s.key})"
    )
    st.caption(spec.blurb)
    st.markdown(
        f"**Input:** `{spec.input_kind}` · scale `{spec.item_scale}`"
        + (f" · {spec.n_items} columns" if spec.input_kind == "matrix" else "")
    )
    st.divider()
    use_llm = st.toggle("LLM narrative", value=True)
    available = llm.available_backends()
    backend = None
    if use_llm:
        if available:
            backend = st.selectbox("Backend", options=available)
            st.caption(f"Model: `{llm.BACKENDS[backend]}`")
        else:
            st.info(
                "No API key detected — set ANTHROPIC_API_KEY, GEMINI_API_KEY or "
                "OPENAI_API_KEY to enable narratives. Scores work fully offline.",
                icon="◇",
            )
    st.divider()
    st.markdown(f"#### All {len(REGISTRY)} metrics")
    overview = ""
    for fam, fam_specs in grouped.items():
        overview += (
            f"<div class='umk-fam'>{family_label(fam)} "
            f"<small>· {len(fam_specs)}</small></div>"
            f"<div class='umk-mlist'>{' · '.join(s.key for s in fam_specs)}</div>"
        )
    st.markdown(overview, unsafe_allow_html=True)

# --- sample data ------------------------------------------------------------
SAMPLES = {
    "sus": "5,1,5,1,5,1,5,1,5,1\n4,2,4,2,4,1,4,2,4,2\n5,1,4,2,5,1,4,1,4,2\n"
           "4,2,3,3,4,2,4,2,3,2\n5,2,5,1,4,1,5,1,5,1\n3,3,4,2,3,2,4,3,4,2",
    "nps": "10\n9\n9\n8\n7\n6\n10\n9\n3\n8",
    "task_success": "1\n1\n1\n0\n1\n1\n0\n1",
    "time_on_task": "42\n38\n55\n61\n29\n47\n88\n40",
    "nasa_tlx": "60,20,50,30,55,40\n70,15,60,25,65,50\n50,10,40,20,45,30",
    "retention_rate": "0.82\n0.61\n0.74\n0.9\n0.55\n0.7\n0.8\n0.66",
    "prompt_iterations": "2\n3\n1\n4\n2\n2\n5\n3",
    "human_fallback": "0\n0\n1\n0\n0\n0\n1\n0\n0\n0",
    "ai_acceptance": "1\n1\n0\n1\n1\n1\n0\n1\n1\n1\n1\n0\n1\n1",
    "trust_automation": "2,2,1,2,1,6,6,5,6,6,6,5\n3,3,2,3,2,5,5,5,6,5,6,5\n"
                        "2,1,2,1,1,6,7,6,6,6,7,6\n3,2,2,3,2,5,6,5,5,6,5,5",
    "poq": "6,6,5\n5,6,6\n6,5,5\n7,6,6\n5,5,6",
    "appropriate_reliance": "1,1\n1,1\n0,0\n1,1\n0,0\n1,1\n0,1\n1,0\n1,1\n0,0",
    "relative_reliance": "0,1,1\n0,1,1\n1,0,0\n0,1,0\n1,0,0\n0,1,1\n1,0,1\n1,0,0",
    "productivity_delta": "82,100\n91,118\n74,96\n88,109\n79,102\n95,121\n70,93\n86,111",
}


def _parse(text: str, kind: str):
    rows = []
    for line in text.strip().splitlines():
        cells = [c.strip() for c in line.replace(",", " ").split() if c.strip()]
        if not cells:
            continue
        try:
            nums = [float(c) for c in cells]
        except ValueError:
            continue  # skip header
        rows.append(nums)
    if kind == "matrix":
        return rows
    flat = [r[0] for r in rows]
    return [int(round(v)) for v in flat] if kind != "values" else flat


col_in, col_out = st.columns([1, 1.3], gap="large")

with col_in:
    st.markdown("#### Data")
    up = st.file_uploader("Upload CSV", type=["csv"], key=spec.key)
    default = SAMPLES.get(spec.key, "")
    if up is not None:
        raw = pd.read_csv(io.BytesIO(up.getvalue()), header=None).to_csv(
            index=False, header=False
        )
        text = st.text_area("Or paste rows", value=raw, height=200)
    else:
        text = st.text_area(
            "Paste rows (one respondent/observation per line)",
            value=default, height=200,
            placeholder="e.g. 5,1,5,1,5,1,5,1,5,1",
        )
    go = st.button("Score metric", use_container_width=True)

with col_out:
    st.markdown("#### Result")
    if go:
        try:
            data = _parse(text, spec.input_kind)
            result = spec.scorer(data)
        except Exception as exc:  # noqa: BLE001 - surface any scoring error to UI
            st.error(f"Could not score: {exc}")
        else:
            payload = result.to_dict()
            if use_llm and llm.is_available(backend):
                with st.spinner("Generating narrative…"):
                    llm.enrich(payload, backend)

            arrow = "▲ higher is better" if result.direction == Direction.HIGHER_BETTER else "▼ lower is better"
            ci = payload.get("interval")
            ci_html = (
                f'<span class="umk-ci">95% CI {ci["low"]}–{ci["high"]} · {ci["method"]}</span>'
                if ci else ""
            )
            chips = ""
            for label in ("grade", "adjective", "acceptability", "band"):
                if label in payload.get("benchmark", {}):
                    cls = "umk-chip-good" if label in ("grade", "band") else ""
                    chips += f'<span class="umk-chip {cls}">{payload["benchmark"][label]}</span>'

            st.markdown(
                f'<div class="umk-card">'
                f'<div class="umk-label">{result.name} · {result.scale} · {arrow}</div>'
                f'<div class="umk-score">{payload["score"]}</div>{ci_html}<br>{chips}'
                f'<div style="margin-top:10px;color:#3a3a37;">{payload["interpretation"]}</div>'
                + (f'<div class="umk-narr">🗒 {payload["llm_summary"]}</div>'
                   if payload.get("llm_summary") else "")
                + "".join(
                    f'<div class="umk-note">⚠ {w}</div>' for w in payload.get("warnings", [])
                )
                + f'<div class="umk-label" style="margin-top:12px;">n = {payload["n"]}</div>'
                "</div>",
                unsafe_allow_html=True,
            )

            comp = payload.get("components", {})
            if isinstance(comp.get("subscales"), dict):
                st.markdown("**Sub-scales**")
                st.bar_chart(pd.Series(comp["subscales"]))
            elif isinstance(comp.get("dimensions"), dict):
                st.markdown("**Workload dimensions**")
                st.bar_chart(pd.Series(comp["dimensions"]))
            elif comp.get("over_reliance") or comp.get("under_reliance"):
                rates = {}
                if comp.get("over_reliance"):
                    rates["over-reliance"] = comp["over_reliance"]["rate"]
                if comp.get("under_reliance"):
                    rates["under-reliance"] = comp["under_reliance"]["rate"]
                st.markdown("**Reliance errors (%)**")
                st.bar_chart(pd.Series(rates))

            with st.expander("Full JSON"):
                st.json(payload)
    else:
        st.caption("Pick a metric, drop in your data, and hit **Score metric**.")

st.divider()
st.caption(
    f"{len(REGISTRY)} metrics · part of the CPR Orbital Series · "
    "rule-based core runs offline; LLM narratives are additive."
)
