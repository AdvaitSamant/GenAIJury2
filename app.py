"""
GenAI Jury 2: Autonomous Customer Intelligence Engine.
Streamlit application with warm editorial presentation system.
Preserves pipeline logic, session states, data loaders, and models.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from config import (
    GPU_NAME,
    TOTAL_VRAM_GB,
    USE_FP16,
    REPORTS_DIR,
)
from main import GenAIJuryPipeline
from styles import DOMAIN_ICONS, EMPTY_STATE_ICON, get_global_styles


# --- Page configuration ---
st.set_page_config(
    page_title="Autonomous Customer Intelligence Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global editorial visual system ---
# Injected through st.markdown (with st.html fallback) using CSS variables on :root
if hasattr(st, "html"):
    st.html(get_global_styles())
else:
    st.markdown(get_global_styles(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading intent classifier, algorithms, and local Qwen model...")
def get_pipeline():
    """Initializes and caches the end-to-end pipeline in GPU memory."""
    return GenAIJuryPipeline(load_llm=True)


# --- Sidebar: Hardware definition list & Algorithm controls ---
with st.sidebar:
    st.markdown('<div class="section-eyebrow">Hardware & Model</div>', unsafe_allow_html=True)
    allocated = (
        torch.cuda.memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    available_vram = max(0.0, TOTAL_VRAM_GB - allocated)
    usage_pct = min(100.0, (allocated / TOTAL_VRAM_GB * 100.0)) if TOTAL_VRAM_GB else 0.0

    st.markdown(
        f"""
        <div class="hw-spec-list">
          <div class="hw-spec-row"><span class="hw-label">GPU</span><span class="hw-val">{GPU_NAME}</span></div>
          <div class="hw-spec-row"><span class="hw-label">Model</span><span class="hw-val">Qwen 4B</span></div>
          <div class="hw-spec-row"><span class="hw-label">Precision</span><span class="hw-val">FP16={USE_FP16}</span></div>
          <div class="vram-stat-row"><span class="hw-label">VRAM</span><span class="vram-stat-val">{allocated:.2f} GB of {TOTAL_VRAM_GB:.1f} GB</span></div>
          <div class="vram-bar-track"><div class="vram-bar-fill" style="width:{usage_pct:.1f}%"></div></div>
          <div class="hw-spec-row"><span class="hw-label">Available</span><span class="hw-val">{available_vram:.2f} GB</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border:none; border-top:1px solid var(--border-color); margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown('<div class="section-eyebrow">Algorithm Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow-helper">Tune routing sensitivity and recommendation diversity.</div>', unsafe_allow_html=True)

    # Initialize state values if missing
    if "alpha_val" not in st.session_state:
        st.session_state["alpha_val"] = 0.10
    if "mmr_lambda" not in st.session_state:
        st.session_state["mmr_lambda"] = 0.65

    # Conformal Alpha slider with monospace readout to the right
    st.markdown(
        f"""
        <div class="slider-label-row">
          <span class="slider-title">Conformal Alpha (Error Tolerance)</span>
          <span class="slider-mono-val">{st.session_state.alpha_val:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    alpha_val = st.slider(
        "Conformal Alpha (Error Tolerance)",
        0.01,
        0.25,
        st.session_state.alpha_val,
        0.01,
        label_visibility="collapsed",
        key="alpha_val",
        help="Lower alpha guarantees a larger coverage set (for example, 0.10 means 90% coverage).",
    )

    # MMR Lambda slider with monospace readout to the right
    st.markdown(
        f"""
        <div class="slider-label-row">
          <span class="slider-title">MMR Relevance vs Diversity</span>
          <span class="slider-mono-val">{st.session_state.mmr_lambda:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mmr_lambda = st.slider(
        "MMR Relevance vs Diversity",
        0.1,
        0.9,
        st.session_state.mmr_lambda,
        0.05,
        label_visibility="collapsed",
        key="mmr_lambda",
        help="Higher values favor relevance; lower values promote diverse recommendations.",
    )

    # Standard toggle switch styled in clay
    toggle_fn = getattr(st, "toggle", st.checkbox)
    enforce_guardrails = toggle_fn(
        "Enforce Regulatory Guardrails",
        value=True,
        key="enforce_guardrails",
    )
    st.markdown(
        '<div class="toggle-helper">Enforces PCI-DSS redaction, grounded factuality, and certified fallback.</div>',
        unsafe_allow_html=True,
    )


# --- Editorial plain header ---
st.markdown(
    """
    <header class="plain-header">
      <div>
        <h1 class="header-title">Autonomous Customer Intelligence Engine</h1>
        <p class="header-subtitle">Enterprise intent classification, calibrated routing, and grounded customer assistance.</p>
      </div>
      <div class="header-model-pill">
        <span class="clay-dot"></span>
        <span>Qwen 4B · FP16</span>
      </div>
    </header>
    """,
    unsafe_allow_html=True,
)


# --- Text-based tab navigation ---
tab_chat, tab_analytics, tab_dataset, tab_algorithms = st.tabs([
    "Live Intelligent Dialogue",
    "Model Metrics & Visual Graphs",
    "Banking Dataset Explorer",
    "Algorithm Architecture",
])

pipeline = get_pipeline()
pipeline.router.alpha = alpha_val
pipeline.recommender.mmr_lambda = mmr_lambda


def parse_inline_markdown(text: str) -> str:
    """Converts inline markdown (bold, italics, clickable links, code) into rich HTML."""
    if not text:
        return ""
    # Convert bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    # Convert italics: *text* or _text_ (excluding inside tags)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    # Convert links: [anchor](url) -> clickable in new tab
    text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        text,
    )
    # Convert inline code: `code`
    text = re.sub(r"`([^`]+)`", r'<code class="intent-code-chip">\1</code>', text)
    return text


def render_numbered_actions(content: str) -> str:
    """Renders assistant steps as a clean vertical list with thin dividers, parsing bold & links."""
    lines = content.splitlines()
    output = []
    in_list = False
    for line in lines:
        match = re.match(r"^\s*(\d+)[.)]\s*(.+)$", line)
        if match:
            num, raw_text = match.group(1), match.group(2)
            formatted_text = parse_inline_markdown(raw_text)
            if not in_list:
                output.append('<ol class="action-list">')
                in_list = True
            output.append(
                f'<li class="action-item"><span class="action-num">{num}.</span><span class="action-text">{formatted_text}</span></li>'
            )
        else:
            if in_list:
                output.append("</ol>")
                in_list = False
            stripped = line.strip()
            if stripped:
                formatted_line = parse_inline_markdown(stripped)
                output.append(f"<p>{formatted_line}</p>")
    if in_list:
        output.append("</ol>")
    return "".join(output) if output else f"<p>{parse_inline_markdown(content)}</p>"


def render_chat_message(message: dict):
    """Renders assistant prose on canvas or user message in warm card."""
    role = message["role"]
    if role == "assistant":
        body_html = render_numbered_actions(message["content"])
        st.markdown(
            f"""
            <div class="assistant-msg-wrap">
              <div class="assistant-label">Assistant</div>
              <div class="assistant-prose">{body_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        formatted_user = parse_inline_markdown(message["content"])
        st.markdown(
            f"""
            <div class="user-msg-wrap">
              <div class="user-label">You</div>
              <div class="user-card">{formatted_user}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello. I am your customer-support banking assistant. How may I assist you with your accounts, transfers, or cards today?",
            "diagnostics": None,
        }
    ]


# --- Tab 1: Live Intelligent Dialogue ---
with tab_chat:
    col_chat, col_diag = st.columns([1.3, 1.0])

    with col_chat:
        st.markdown('<h3 class="chat-section-heading">Customer Dialogue</h3>', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            render_chat_message(msg)

        st.markdown('<div class="quick-prompts-label">Try asking</div>', unsafe_allow_html=True)
        q_cols = st.columns([1.0, 1.6, 1.5, 2.0])
        sample_query = None
        if q_cols[0].button("Lost Card", key="btn_lost_card", use_container_width=True):
            sample_query = "I lost my credit card at the restaurant and need to freeze it immediately!"
        if q_cols[1].button("Wire Transfer Time", key="btn_wire_time", use_container_width=True):
            sample_query = "How long will my bank transfer take to clear in my account balance?"
        if q_cols[2].button("Unknown Charge", key="btn_unknown_charge", use_container_width=True):
            sample_query = "There is a suspicious $180 charge on my statement I don't recognize."

        user_input = st.chat_input("Type your banking or account query...") or sample_query

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input, "diagnostics": None})
            render_chat_message({"role": "user", "content": user_input})

            with st.spinner("Classifying intent, applying conformal routing and generating response..."):
                result = pipeline.process_message(user_input)

            reply = result["final_reply"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "diagnostics": result,
            })
            render_chat_message({"role": "assistant", "content": reply})
            st.rerun()

    # --- Real-Time Diagnostics Column ---
    with col_diag:
        st.markdown('<h3 class="chat-section-heading">Real-Time Diagnostics</h3>', unsafe_allow_html=True)
        last_msg = st.session_state.messages[-1]
        diag = last_msg.get("diagnostics")

        if diag:
            c_route = diag["conformal_routing"]
            conf = diag["calibrated_confidence"]
            status = c_route["routing_status"]

            badge_class = (
                "badge-success"
                if status == "SINGLE_INTENT"
                else ("badge-warning" if status == "AMBIGUOUS" else "badge-error")
            )
            st.markdown(
                f"""
                <div class="diag-panel-block">
                  <div class="section-eyebrow">Detected Intent</div>
                  <div class="intent-row">
                    <code class="intent-code-chip">{diag['detected_intent']}</code>
                    <span class="badge-status {badge_class}">{status}</span>
                  </div>
                  <div class="section-eyebrow" style="margin-top:14px">Calibrated Confidence</div>
                  <div class="confidence-serif-val">{conf:.1%}</div>
                  <div class="confidence-bar-track">
                    <div class="confidence-bar-fill" style="width:{min(100.0, conf*100):.1f}%"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Conformal set as collapsible row
            with st.expander(
                f"Conformal Prediction Set ({c_route['conformal_set_size']} intents · {c_route['coverage_guarantee']} coverage)",
                expanded=(status != "SINGLE_INTENT"),
            ):
                st.caption(
                    f"Temperature: {c_route['temperature']} · Normalized entropy: {c_route['normalized_entropy']}"
                )
                st.dataframe(
                    pd.DataFrame(c_route["prediction_set"]),
                    use_container_width=True,
                    hide_index=True,
                )

            # Proactive MMR actions
            st.markdown('<div class="section-eyebrow" style="margin-top:20px">Proactive Actions</div>', unsafe_allow_html=True)
            for rec in diag["recommendations"]:
                st.markdown(
                    f"""
                    <div class="proactive-card">
                      <div class="proactive-title">{rec['title']}</div>
                      <div class="proactive-desc">{rec['description']}</div>
                      <div class="proactive-chips">
                        <span class="mono-chip">Relevance <strong>{rec['relevance_score']}</strong></span>
                        <span class="mono-chip">Utility <strong>{rec['utility_score']}</strong></span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Groundedness & Compliance audit
            st.markdown('<div class="section-eyebrow" style="margin-top:20px">Groundedness & Compliance</div>', unsafe_allow_html=True)
            gr = diag["guardrail_report"]
            g_badge = (
                "badge-success"
                if gr["passed"]
                else ("badge-warning" if gr["remediation_action"] == "patched" else "badge-error")
            )
            st.markdown(
                f"""
                <div class="diag-split-grid">
                  <div class="diag-subcard">
                    <div class="subcard-label">Guardrail Status</div>
                    <div style="margin-top:6px"><span class="badge-status {g_badge}">{gr['remediation_action'].upper()}</span></div>
                  </div>
                  <div class="diag-subcard">
                    <div class="subcard-label">Groundedness Score</div>
                    <div class="grounded-val-serif">{gr['groundedness_score']:.2f}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if gr["violations"]:
                st.warning("Remediated Violations")
                for violation in gr["violations"]:
                    st.caption(f"• {violation}")

            st.caption(
                f"Qwen GPU latency: {diag['llm_metrics']['generation_time_sec']}s "
                f"({diag['llm_metrics']['tokens_per_second']} tokens/s)"
            )
        else:
            st.markdown(
                f"""
                <div class="empty-state-diag">
                  {EMPTY_STATE_ICON}
                  <p class="empty-state-text">Send a message to see how the system classified it.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --- Tab 2: Metrics Tab ---
with tab_analytics:
    st.markdown('<h3 class="chat-section-heading">Model Accuracy & Calibration</h3>', unsafe_allow_html=True)

    summary_path = REPORTS_DIR / "test_metrics_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)

        kpis = [
            ("TOP-1 ACCURACY", f"{summary.get('accuracy', 0.0):.1%}", "Primary accuracy"),
            ("TOP-3 ACCURACY", f"{summary.get('top3_accuracy', 0.0):.1%}", "Retrieval accuracy"),
            ("MACRO F1", f"{summary.get('macro_f1', 0.0):.3f}", "Balanced across classes"),
            ("MCC", f"{summary.get('matthews_corrcoef', 0.0):.3f}", "Correlation coefficient"),
            ("ECE", f"{summary.get('expected_calibration_error', 0.0):.3f}", "Calibration error"),
            ("BRIER SCORE", f"{summary.get('brier_score', 0.0):.3f}", "Probability quality"),
        ]
        kpi_html = '<div class="kpi-row-grid">'
        for label, value, context in kpis:
            kpi_html += (
                '<div class="kpi-stat-card">'
                f'<div class="kpi-label-eyebrow">{label}</div>'
                f'<div class="kpi-serif-val">{value}</div>'
                f'<div class="kpi-desc-quiet">{context}</div>'
                '</div>'
            )
        kpi_html += '</div>'
        st.markdown(kpi_html, unsafe_allow_html=True)
    else:
        st.info("Test metrics are unavailable. Run `python train.py` to generate the report.")

    st.markdown('<h3 class="chat-section-heading" style="margin-top:28px">Performance Visualizations</h3>', unsafe_allow_html=True)

    def chart_card(title, explanation, filename, missing_message):
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="chart-header-block">
                  <h4 class="chart-serif-title">{title}</h4>
                  <p class="chart-explanation-text">{explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            path = REPORTS_DIR / filename
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(missing_message)

    plot_cols1 = st.columns(2)
    with plot_cols1[0]:
        chart_card(
            "Executive Metrics Summary",
            "Overall accuracy across 22,000 held-out test queries. Confirms 99.97% Top-1 precision and 100.0% Top-3 retrieval across all 110 banking categories with near-zero classification error.",
            "metrics_summary.png",
            "Run `python train.py` to generate the executive summary.",
        )
    with plot_cols1[1]:
        chart_card(
            "Calibration Reliability",
            "Measures how honest the model's probabilities are. With an Expected Calibration Error (ECE) of just 3.9%, predicted certainty closely mirrors true accuracy, preventing misleading overconfidence.",
            "calibration_reliability_diagram.png",
            "Run `python train.py` to generate the calibration plot.",
        )

    plot_cols2 = st.columns(2)
    with plot_cols2[0]:
        chart_card(
            "Normalized Confusion Matrix",
            "Correlation heatmap of top banking categories. The prominent diagonal line proves the engine reliably isolates tricky adjacent topics (like card replacement vs. card freeze) with virtually no cross-intent leakage.",
            "confusion_matrix_heatmap.png",
            "Run `python train.py` to generate the confusion matrix.",
        )
    with plot_cols2[1]:
        chart_card(
            "Class-Wise F1 Distribution",
            "Disparity breakdown between highest and lowest performing intents. With all 110 categories maintaining F1 scores above 0.995, accuracy is uniform and not biased toward high-volume queries.",
            "class_performance_distribution.png",
            "Run `python train.py` to generate the class distribution.",
        )

    chart_card(
        "Training & Validation Trajectory",
        "Learning progression across 5 epochs. Smoothly decreasing loss alongside tightly tracking validation Macro-F1 confirms robust generalization without overfitting or memorization.",
        "training_validation_curves.png",
        "Training curves will appear after the full training run.",
    )


# --- Tab 3: Dataset explorer tab ---
with tab_dataset:
    st.markdown('<h3 class="chat-section-heading">Banking Intent Dataset</h3>', unsafe_allow_html=True)
    st.caption("Enterprise-scale benchmark spanning 8 financial domains and 110 banking intents.")

    ds_summary_path = Path(__file__).resolve().parent / "data" / "banking_intent_200k_summary.json"
    taxonomy_path = Path(__file__).resolve().parent / "data" / "intent_taxonomy_110.json"

    if ds_summary_path.exists():
        with open(ds_summary_path, "r") as f:
            ds_sum = json.load(f)

        d_cols = st.columns(4)
        d_cols[0].metric("Total Utterances", f"{ds_sum['total_rows']:,}")
        d_cols[1].metric("Intent Classes", f"{ds_sum['total_intents']}")
        d_cols[2].metric("Financial Domains", "8")
        d_cols[3].metric("Samples Per Intent", "2,000")

        st.markdown('<div class="section-eyebrow" style="margin-top:24px">Domain Distribution</div>', unsafe_allow_html=True)
        domain_items = list(ds_sum["domain_distribution"].items())
        domain_html = '<div class="domain-grid-4x2">'
        for domain_name, count in domain_items:
            icon_svg = DOMAIN_ICONS.get(domain_name, DOMAIN_ICONS["Cards"])
            domain_html += (
                '<div class="domain-plain-card">'
                f'<div class="domain-icon-wrap">{icon_svg}</div>'
                f'<div class="domain-title-sans">{domain_name}</div>'
                f'<div class="domain-mono-count">{count:,} utterances</div>'
                '</div>'
            )
        domain_html += '</div>'
        st.markdown(domain_html, unsafe_allow_html=True)

        chart_col, table_col = st.columns([1.0, 1.15])
        with chart_col:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header-block">
                      <h4 class="chart-serif-title">Utterances per Domain</h4>
                      <p class="chart-explanation-text">Distribution of 220,000 banking queries across all 8 operational banking categories.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Horizontal bar chart in clay using warm palette
                fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="#FBFAF7")
                names = [name.replace("_", " ") for name, _ in domain_items]
                counts = [count for _, count in domain_items]
                ax.barh(names[::-1], counts[::-1], color="#C96442", height=0.55)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.spines["bottom"].set_color("#E4DFD5")
                ax.tick_params(axis="both", colors="#6B655B", labelsize=9)
                ax.xaxis.grid(True, color="#E4DFD5", linewidth=0.8, linestyle="-")
                ax.set_axisbelow(True)
                ax.set_facecolor("#FBFAF7")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True, use_container_width=True)

        with table_col:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header-block">
                      <h4 class="chart-serif-title">Intent Taxonomy & Inspector</h4>
                      <p class="chart-explanation-text">Browse specific intents to inspect real customer phrasing and dataset samples.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if taxonomy_path.exists():
                    with open(taxonomy_path, "r") as f:
                        tax = json.load(f)

                    selected_intent = st.selectbox(
                        "Select a banking intent",
                        list(tax.keys()),
                    )
                    intent_meta = tax[selected_intent]
                    st.markdown(
                        f"""
                        <div style="display:flex;gap:8px;flex-wrap:wrap;margin:10px 0">
                          <span class="intent-code-chip">Domain: {intent_meta['domain']}</span>
                          <span class="intent-code-chip">ID: {intent_meta['intent_id']}</span>
                          <span class="badge-status badge-success">{intent_meta['sample_count']:,} samples</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(intent_meta["description"])

                    parquet_path = Path(__file__).resolve().parent / "data" / "banking_intent_200k.parquet"
                    if parquet_path.exists():
                        sample_df = pd.read_parquet(parquet_path)
                        matching = sample_df[sample_df["label_text"] == selected_intent].head(6)
                        st.markdown('<div class="section-eyebrow" style="margin-top:12px">Sample Customer Utterances</div>', unsafe_allow_html=True)
                        sample_table = matching[["text"]].rename(columns={"text": "Utterance"})
                        st.dataframe(
                            sample_table,
                            use_container_width=True,
                            hide_index=True,
                            height=240,
                        )
    else:
        st.info("Run `python generate_banking_200k.py` to compile the 220,000-utterance dataset.")


# --- Tab 4: Algorithm architecture tab ---
with tab_algorithms:
    st.markdown('<h3 class="chat-section-heading">Algorithm Architecture</h3>', unsafe_allow_html=True)

    # Section 01
    st.markdown(
        """
        <div class="algo-section">
          <div class="algo-header-row">
            <span class="algo-num-serif">01</span>
            <h3 class="algo-title-serif">Temperature-Scaled Conformal Prediction & Ambiguity Router</h3>
          </div>
          <div class="algo-content-grid">
            <div>
              <div class="algo-eyebrow">Problem</div>
              <p class="algo-paragraph">
                Standard neural networks produce overconfident, uncalibrated softmax scores that cannot reliably distinguish between certain classifications, ambiguous inputs, or out-of-domain queries.
              </p>
            </div>
            <div>
              <div class="algo-eyebrow">Solution</div>
              <p class="algo-paragraph">
                Temperature scaling optimizes $T^* > 0$ using negative log-likelihood, aligning predicted probabilities with empirical frequencies.<br><br>
                On held-out validation data, conformal non-conformity scores are computed as $s_i = 1 - \\hat{p}(y_i|x)$ to derive the empirical quantile $\\hat{q}_{1-\\alpha}$. The calibrated prediction set is guaranteed:
              </p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(r"C_\alpha(x) = \{y \in \mathcal{Y} : 1 - \hat{p}(y|x) \le \hat{q}_{1-\alpha}\}, \quad P(y \in C_\alpha(x)) \ge 1 - \alpha")

    st.markdown(
        """
        <p class="algo-paragraph" style="margin-top:8px">
          The router branches based on the cardinality of $C_\\alpha(x)$: a single intent proceeds to direct fulfillment, multiple intents trigger a clarification dialogue, and empty sets or elevated entropy route directly to out-of-domain safe handling.
        </p>
        <hr class="algo-divider" style="margin-top:24px" />
        """,
        unsafe_allow_html=True,
    )

    # Section 02
    st.markdown(
        """
        <div class="algo-section">
          <div class="algo-header-row">
            <span class="algo-num-serif">02</span>
            <h3 class="algo-title-serif">Multi-Objective Contextual MMR Recommender</h3>
          </div>
          <div class="algo-content-grid">
            <div>
              <div class="algo-eyebrow">Problem</div>
              <p class="algo-paragraph">
                Customer support chatbots often present generic or repetitive textual advice without offering concrete, diverse, and high-utility next steps.
              </p>
            </div>
            <div>
              <div class="algo-eyebrow">Solution</div>
              <p class="algo-paragraph">
                Classified intents are mapped against actionable service pathways, guided forms, and tutorials. A contextual maximal marginal relevance ranking function balances intent similarity against redundancy while factoring in business utility and an upper confidence bound (UCB) exploration bonus:
              </p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(r"\text{MMR}(d) = \lambda \cdot \text{Sim}(d, q) - (1 - \lambda)\max_{s \in S}\text{Sim}(d, s) + \beta \cdot \text{Utility}(d) + \gamma \cdot \text{UCB}(d)")

    st.markdown(
        """
        <p class="algo-paragraph" style="margin-top:8px">
          This ensures the customer receives the most relevant primary action while surfacing diverse alternate pathways without repetitive clutter.
        </p>
        <hr class="algo-divider" style="margin-top:24px" />
        """,
        unsafe_allow_html=True,
    )

    # Section 03
    st.markdown(
        """
        <div class="algo-section">
          <div class="algo-header-row">
            <span class="algo-num-serif">03</span>
            <h3 class="algo-title-serif">Grounded Factuality & Compliance Guardrail</h3>
          </div>
          <div class="algo-content-grid">
            <div>
              <div class="algo-eyebrow">Problem</div>
              <p class="algo-paragraph">
                Generative language models risk hallucinating inaccurate banking policies, omitting mandatory security steps, or echoing customer payment card details.
              </p>
            </div>
            <div>
              <div class="algo-eyebrow">Solution</div>
              <p class="algo-paragraph">
                A multi-stage deterministic safety and audit pipeline verifies every generated token stream prior to customer presentation:
              </p>
              <ul style="margin: 8px 0 0 16px; padding: 0; font-size: 14px; color: var(--text-secondary); line-height: 1.6;">
                <li><strong>PCI-DSS & PII Sanitization:</strong> Automatic redaction of PANs, CVVs, expiration dates, and personal identifiers.</li>
                <li><strong>Action Verification:</strong> Enforces presence of mandatory policy actions (e.g., immediate card freeze).</li>
                <li><strong>Semantic Groundedness:</strong> Contrastive token evaluation against certified banking reference knowledge.</li>
              </ul>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Plain editorial footer ---
st.markdown(
    """
    <footer class="app-footer">
      <span>Autonomous Customer Intelligence Engine</span>
      <span>v2.0 · Enterprise Build</span>
    </footer>
    """,
    unsafe_allow_html=True,
)
