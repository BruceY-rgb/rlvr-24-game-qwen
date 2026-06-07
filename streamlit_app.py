from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import streamlit as st

from twentyfour_rl.data import read_jsonl
from twentyfour_rl.eval import generate_completion, load_model_and_tokenizer, solver_completion
from twentyfour_rl.rewards import score_output
from twentyfour_rl.verifier import extract_answer, verify


st.set_page_config(
    page_title="24 Game RLVR Lab",
    page_icon="24",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
  :root {
    --ink: #172018;
    --muted: #536257;
    --paper: #f6f1e8;
    --panel: #fffaf1;
    --sidebar: #eee4d3;
    --line: #cfc2aa;
    --line-strong: #a99d88;
    --accent: #c94135;
    --accent-deep: #9b2e27;
    --green: #1f6b4a;
    --field: #fffdf7;
  }
  .stApp {
    background: var(--paper);
    color: var(--ink);
  }
  .stApp,
  .stApp p,
  .stApp span,
  .stApp label,
  .stApp div,
  [data-testid="stMarkdownContainer"] {
    color: var(--ink);
  }
  [data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--line);
  }
  [data-testid="stSidebar"] *,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span {
    color: var(--ink) !important;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: var(--ink) !important;
  }
  h1, h2, h3 {
    color: var(--ink);
    letter-spacing: 0;
  }
  button[data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 800;
    opacity: 1;
  }
  button[data-baseweb="tab"] p,
  button[data-baseweb="tab"] div {
    color: inherit !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
  }
  [data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
  }
  input,
  textarea,
  [data-baseweb="input"] input,
  [data-baseweb="textarea"] textarea,
  [data-baseweb="select"] > div {
    background: var(--field) !important;
    color: var(--ink) !important;
    border-color: var(--line-strong) !important;
    box-shadow: none !important;
  }
  input::placeholder,
  textarea::placeholder {
    color: #7b857b !important;
    opacity: 1 !important;
  }
  [data-baseweb="select"] svg {
    color: var(--ink) !important;
    fill: var(--ink) !important;
  }
  [data-baseweb="popover"],
  [data-baseweb="popover"] * {
    background: var(--field) !important;
    color: var(--ink) !important;
  }
  .stButton > button {
    background: var(--accent) !important;
    color: #fffaf1 !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
  }
  .stButton > button * {
    color: #fffaf1 !important;
  }
  [data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: .8rem .95rem;
  }
  [data-testid="stMetricLabel"] *,
  [data-testid="stMetricValue"] *,
  [data-testid="stMetricDelta"] * {
    color: var(--ink) !important;
  }
  [data-testid="stMetricLabel"] * {
    color: var(--muted) !important;
    font-size: .92rem !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricValue"] * {
    font-size: clamp(1.4rem, 2.4vw, 2.15rem) !important;
    line-height: 1.05 !important;
    white-space: nowrap !important;
  }
  .lab-title {
    font-size: clamp(2rem, 5vw, 4.6rem);
    line-height: .92;
    font-weight: 800;
    max-width: 980px;
    margin-bottom: .25rem;
  }
  .lab-subtitle {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 900px;
  }
  .status-strip {
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: .8rem 0;
    margin: 1.2rem 0 1.5rem 0;
    color: var(--muted);
  }
  .verdict-ok {
    color: var(--green);
    font-weight: 800;
  }
  .verdict-bad {
    color: var(--accent-deep);
    font-weight: 800;
  }
  .mono-block {
    background: var(--panel);
    border: 1px solid var(--line);
    color: var(--ink);
    padding: 1rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    border-radius: 8px;
    line-height: 1.55;
  }
  .mono-block,
  .mono-block * {
    color: var(--ink) !important;
  }
  code {
    color: var(--accent-deep) !important;
    background: #efe6d7 !important;
    border-radius: 4px;
  }
</style>
"""


st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_jsonl_cached(path: str) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    return read_jsonl(path)


@st.cache_data(show_spinner=False)
def load_metrics_json(path: str) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=True)
def load_online_model(model_path: str, adapter_path: str | None, dtype: str):
    return load_model_and_tokenizer(model_path, adapter_path or None, dtype=dtype)


def first_matching_offline(rows: list[dict[str, Any]], numbers: list[int]) -> dict[str, Any] | None:
    wanted = sorted(numbers)
    for row in rows:
        if sorted(int(n) for n in row.get("numbers", [])) == wanted:
            return row
    return rows[0] if rows else None


def dataframe_from_jsonl(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def render_verdict(scored: dict[str, Any]) -> None:
    verdict = "正确" if scored["is_correct"] else "未通过"
    css_class = "verdict-ok" if scored["is_correct"] else "verdict-bad"
    st.markdown(f"<div class='{css_class}'>{verdict} · {scored['error_type']}</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Reward", f"{scored['reward']:.2f}")
    cols[1].metric("Value", "NA" if scored["value"] is None else f"{scored['value']:.6g}")
    cols[2].metric("Legal", "Yes" if scored["is_valid"] else "No")
    cols[3].metric("Format", "Yes" if scored["format"] else "No")
    if scored.get("message"):
        st.caption(scored["message"])


def render_single_case(raw_output: str, numbers: list[int], target: int) -> None:
    scored = score_output(numbers, raw_output, target)
    answer = extract_answer(raw_output)
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("模型输出")
        st.markdown(f"<div class='mono-block'>{raw_output}</div>", unsafe_allow_html=True)
        st.subheader("Answer")
        st.markdown(f"<div class='mono-block'>{answer or '(empty)'}</div>", unsafe_allow_html=True)
    with right:
        st.subheader("程序判定")
        render_verdict(scored)
        steps = scored.get("calculation_steps") or []
        if steps:
            st.subheader("计算轨迹")
            st.markdown("\n".join(f"- `{step}`" for step in steps))


def chart_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


st.sidebar.header("运行配置")
mode = st.sidebar.selectbox("推理模式", ["离线结果回放", "精确搜索基线", "在线模型推理"])
eval_path = st.sidebar.text_input("eval_results.jsonl", "outputs/eval_results.jsonl")
metrics_path = st.sidebar.text_input("eval_metrics.json", "outputs/eval_metrics.json")
train_log_path = st.sidebar.text_input("train_metrics.jsonl", "outputs/qwen24-grpo/train_metrics.jsonl")
model_path = st.sidebar.text_input("模型路径", "Qwen/Qwen2.5-1.5B-Instruct")
adapter_path = st.sidebar.text_input("LoRA adapter 路径", "outputs/qwen24-grpo")
dtype = st.sidebar.selectbox("dtype", ["bf16", "fp16", "fp32"])


st.markdown("<div class='lab-title'>24 Game RLVR Lab</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='lab-subtitle'>用可验证奖励观察模型是否真的学会 24 点推理：答案、格式、数字使用和算术结果都由程序判定。</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='status-strip'>ModelArts Ascend · Qwen2.5-1.5B-Instruct · GRPO / grouped RL fallback · Streamlit evidence board</div>", unsafe_allow_html=True)

offline_rows = load_jsonl_cached(eval_path)
metrics_rows = load_metrics_json(metrics_path)
train_rows = load_jsonl_cached(train_log_path)

tab_solve, tab_train, tab_eval, tab_errors = st.tabs(["单题求解", "训练曲线", "模型对比", "失败分析"])


with tab_solve:
    st.header("单题求解")
    cols = st.columns(5)
    numbers = [
        int(cols[0].number_input("数 1", min_value=1, max_value=13, value=3, step=1)),
        int(cols[1].number_input("数 2", min_value=1, max_value=13, value=3, step=1)),
        int(cols[2].number_input("数 3", min_value=1, max_value=13, value=8, step=1)),
        int(cols[3].number_input("数 4", min_value=1, max_value=13, value=8, step=1)),
    ]
    target = int(cols[4].number_input("目标", min_value=1, max_value=100, value=24, step=1))
    max_new_tokens = st.slider("生成长度", min_value=64, max_value=512, value=256, step=32)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)

    if st.button("运行判定", type="primary"):
        if mode == "精确搜索基线":
            raw = solver_completion(numbers, target)
        elif mode == "离线结果回放":
            row = first_matching_offline(offline_rows, numbers)
            raw = row.get("raw_output", "") if row else ""
            if row is None:
                st.warning("没有找到离线结果，请切换到精确搜索基线或在线模型推理。")
        else:
            model, tokenizer, device = load_online_model(model_path, adapter_path, dtype)
            raw = generate_completion(
                model,
                tokenizer,
                device,
                numbers,
                target,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
            )
        render_single_case(raw, numbers, target)
    else:
        st.caption("输入四个数后运行判定。离线模式会从 eval_results.jsonl 中找同一组数字。")


with tab_train:
    st.header("训练曲线")
    df = dataframe_from_jsonl(train_rows)
    if df.empty:
        st.info("还没有训练日志。运行 train_grpo.py 后加载 train_metrics.jsonl。")
    else:
        df = df.sort_values("step")
        columns = chart_columns(
            df,
            [
                "train_reward",
                "dev_reward",
                "train_exact_rate",
                "dev_exact_rate",
                "train_legal_rate",
                "dev_legal_rate",
                "train_format_rate",
                "dev_format_rate",
                "train_invalid_rate",
                "dev_invalid_rate",
            ],
        )
        st.line_chart(df.set_index("step")[columns])
        length_cols = chart_columns(df, ["completion_length", "train_loss"])
        if length_cols:
            st.line_chart(df.set_index("step")[length_cols])
        st.dataframe(df.tail(20), width="stretch")


with tab_eval:
    st.header("模型对比")
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
    else:
        eval_df = dataframe_from_jsonl(offline_rows)
        if eval_df.empty:
            metrics_df = pd.DataFrame()
        else:
            from twentyfour_rl.eval import aggregate_metrics

            metrics_df = pd.DataFrame(aggregate_metrics(offline_rows))

    if metrics_df.empty:
        st.info("还没有评估结果。运行 eval.py 后加载 eval_metrics.json。")
    else:
        st.dataframe(metrics_df, width="stretch")
        numeric_cols = [
            c
            for c in ["pass_at_1", "pass_at_k", "legal_rate", "format_rate", "invalid_rate", "hallucination_rate"]
            if c in metrics_df.columns
        ]
        if numeric_cols:
            chart_df = metrics_df[["model_name", "split", *numeric_cols]].melt(
                id_vars=["model_name", "split"],
                value_vars=numeric_cols,
                var_name="metric",
                value_name="value",
            )
            chart_df["run"] = chart_df["model_name"].astype(str) + " / " + chart_df["split"].astype(str)
            fig = px.bar(
                chart_df,
                x="run",
                y="value",
                color="metric",
                barmode="group",
                labels={"run": "模型 / 数据集", "value": "比例", "metric": "指标"},
                range_y=[0, 1],
            )
            fig.update_layout(
                margin={"l": 0, "r": 0, "t": 18, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")


with tab_errors:
    st.header("失败分析")
    eval_df = dataframe_from_jsonl(offline_rows)
    if eval_df.empty:
        st.info("还没有 eval_results.jsonl。")
    else:
        error_counts = eval_df.groupby("error_type").size().reset_index(name="count").sort_values("count", ascending=False)
        left, right = st.columns([0.8, 1.2])
        with left:
            fig = px.bar(
                error_counts,
                x="error_type",
                y="count",
                labels={"error_type": "错误类型", "count": "数量"},
            )
            fig.update_layout(
                margin={"l": 0, "r": 0, "t": 18, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")
        with right:
            selected_error = st.selectbox("错误类型", error_counts["error_type"].tolist())
            examples = eval_df[eval_df["error_type"] == selected_error].head(10)
            for _, row in examples.iterrows():
                nums = [int(n) for n in row["numbers"]]
                st.write(f"Numbers: `{nums}` · Model: `{row.get('model_name', 'model')}`")
                st.markdown(f"<div class='mono-block'>{row.get('raw_output', '')}</div>", unsafe_allow_html=True)
                verdict = verify(nums, row.get("answer", ""), target=int(row.get("target", 24)))
                st.caption(verdict["message"])
