import os
import random
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import streamlit as st
from groq import Groq

# ----------------------------------------------------
# 1. Page Configuration & Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="Audit Performance Evaluation - Outcome Effect Simulation",
    page_icon="⚖️️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">⚖️ Auditor Performance Evaluation & Outcome Effect Simulation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Simulating performance evaluations of a staff auditor under different outcome conditions (No Misstatement vs. Misstatement).</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# 2. API Setup
# ----------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "⚠️ API Key is missing! Please configure 'GROQ_API_KEY' in Streamlit secrets."
    )
    st.stop()

client = Groq(api_key=api_key)

# ----------------------------------------------------
# 3. Sidebar Setup
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration Settings")

    num_personas = st.number_input(
        "Number of Replications (Personas/Scenario):",
        min_value=1,
        max_value=500,
        value=25,
        step=1,
    )

    model_name = st.selectbox(
        "Select LLM Model:",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
        ],
    )

    csv_path = st.text_input(
        "Output CSV File Name:", value="outcome_effect_results.csv"
    )

# Load existing CSV into session_state if available
if "df_data" not in st.session_state and os.path.exists(csv_path):
    try:
        st.session_state["df_data"] = pd.read_csv(csv_path)
    except Exception:
        pass

# ----------------------------------------------------
# 4. Experimental Scenarios (Tabs)
# ----------------------------------------------------
st.header("📝 1. Experimental Setup")

tab1, tab2 = st.tabs(["📋 Base Scenario Context", "🔬 Outcome Treatment Scenarios"])

with tab1:
    base_scenario_text = st.text_area(
        "Edit Base Scenario Context:",
        value="""You are the Lead Audit Senior on the year-end audit engagement for Madison, Inc. You are responsible for supervising the audit staff and evaluating their performance at the end of the engagement.
Madison, Inc. is a large, publicly traded manufacturing corporation with multiple operating divisions. Madison has been a major client of your office for 10 years and has consistently received an unqualified (clean) audit opinion every year. Historical audit adjustments have been rare and immaterial. The audit budget for this fiscal year is extremely tight, with heavy pressure from firm leadership to keep audit fees down. Madison's management expects the audit to run smoothly without disruptions. They demand formal explanations whenever the nature, timing, or extent of your audit procedures changes.
You are supervising Sam, a third-year staff auditor. Sam was assigned to perform substantive analytical procedures on the revenue account for Madison’s Sporting Goods Division. Historically, the audit team relied only on Madison's prior-year financial data and general industry financial trends to verify revenue. This year, those financial metrics matched management's numbers perfectly. If Sam had simply repeated the prior year’s steps, the revenue account would have looked entirely correct. Instead of only looking for confirming evidence, Sam incorporated Non-Financial Measures (NFMs)—specifically tracking the number of employees and the square footage of production facilities—which prior audits never evaluated. By comparing the financials to the NFMs, Sam identified a major inconsistency: revenue had increased significantly, but headcount and square footage had decreased.
Driven by professional skepticism, Sam launched an extended investigation into the inconsistency. The investigation pushed Sam over budget and strained relations with management.""",
        height=220,
    )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        no_misstatement_text = st.text_area(
            "Scenario 1: No Misstatement Condition",
            value="""Sam found that the inconsistency described above resulted from the Sporting Goods division outsourcing some operations overseas. After making several inquiries and gathering additional audit evidence, Sam concluded that this revenue account was not misstated.""",
            height=150,
        )
    with col2:
        misstatement_text = st.text_area(
            "Scenario 2: Misstatement Condition",
            value="""Sam discovered that the inconsistency described above resulted from the Sporting Goods division outsourcing some operations overseas. After making several inquiries and gathering additional audit evidence, Sam concluded that the revenue account was significantly misstated because the overseas operation was recognising revenue prematurely.""",
            height=150,
        )

scenarios_dict = {
    "No Misstatement": no_misstatement_text,
    "Misstatement": misstatement_text,
}

# ----------------------------------------------------
# 5. Simulation Execution & Data Download
# ----------------------------------------------------
st.divider()
st.header("🚀 2. Run Simulation")

c1, c2, c3 = st.columns([1.2, 1.2, 2])
with c1:
    run_btn = st.button(
        "▶️ Start Simulation", type="primary", use_container_width=True
    )

with c2:
    if "df_data" in st.session_state and not st.session_state["df_data"].empty:
        csv_bytes = (
            st.session_state["df_data"].to_csv(index=False).encode("utf-8")
        )
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_bytes,
            file_name=csv_path,
            mime="text/csv",
            use_container_width=True,
        )

if run_btn:
    try:
        results = []
        global_persona_id = 1

        positions = ["Lead Audit Senior", "Manager"]
        experiences = ["3 years", "7 years", "11 years", "15 years"]
        genders = ["Male", "Female"]

        total_tasks = num_personas * len(scenarios_dict)
        progress_bar = st.progress(0)
        status_text = st.empty()

        completed = 0
        p_selected = positions[0]

        for i in range(num_personas):
            exp_selected = random.choice(experiences)
            gen_selected = random.choice(genders)

            for sc_name, sc_content in scenarios_dict.items():
                persona_id = f"AP_{global_persona_id}"
                global_persona_id += 1

                system_prompt = (
                    f"You are an auditor with a {p_selected} position. "
                    f"Your experience is {exp_selected} and your gender is {gen_selected}. "
                    f"Always respond strictly with a numeric value as requested."
                )

                user_instruction = f"""
                Background Context: {base_scenario_text}
                Outcome Condition: {sc_content}

                Based on the information presented, how would you evaluate the auditor's overall performance?
                Evaluate performance on an 11-point scale ranging from -5 to +5, where:
                -5 = Below Expectations
                 0 = Met Expectations
                +5 = Above Expectations

                IMPORTANT: Output ONLY a single integer or decimal number between -5 and 5 (e.g., "3" or "-1"). Do not include any extra text.
                """

                status_text.text(
                    f"🔄 Processing {persona_id} | Outcome: {sc_name}..."
                )

                exp_score = None
                max_retries = 3

                for attempt in range(max_retries):
                    try:
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_instruction},
                            ],
                            temperature=0.3,
                        )

                        response_text = (
                            completion.choices[0].message.content.strip()
                        )
                        # Extract integers/floats including negative numbers
                        numbers = re.findall(
                            r"[-+]?\d+(?:\.\d+)?", response_text
                        )

                        if len(numbers) >= 1:
                            val = float(numbers[0])
                            # Ensure within scale limits [-5, 5]
                            exp_score = max(-5.0, min(5.0, val))

                        break

                    except Exception as err:
                        if "429" in str(err) and attempt < max_retries - 1:
                            time.sleep(3)
                        else:
                            break

                time.sleep(0.4)

                results.append({
                    "scenario_type": sc_name,
                    "ID": persona_id,
                    "position": p_selected,
                    "Experience": exp_selected,
                    "gender": gen_selected,
                    "Expectation_Score": exp_score,
                })

                completed += 1
                progress_bar.progress(completed / total_tasks)

        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_path, index=False)
        st.session_state["df_data"] = df_res

        status_text.empty()
        st.success(f"✅ Simulation completed! Generated {len(df_res)} records.")
        st.rerun()

    except Exception as e:
        st.error(f"Execution Error: {e}")

# Raw dataset viewer
if "df_data" in st.session_state and not st.session_state["df_data"].empty:
    with st.expander("🔍 View Raw Simulation Dataset", expanded=False):
        st.dataframe(st.session_state["df_data"], use_container_width=True)

# ----------------------------------------------------
# 6. Analysis Modules
# ----------------------------------------------------
if "df_data" in st.session_state and not st.session_state["df_data"].empty:
    df = st.session_state["df_data"].copy()
    df["Expectation_Score"] = pd.to_numeric(
        df["Expectation_Score"], errors="coerce"
    )
    df_clean = df.dropna(subset=["Expectation_Score"])

    st.divider()
    st.header("📊 3. Analysis & Visualizations")

    # ------------------ Expander 1: Descriptive Stats ------------------
    with st.expander("📈 Descriptive Statistics Analysis", expanded=False):

        def get_enhanced_stats(data, metric_col):
            stats_list = []
            for sc_name, group in data.groupby("scenario_type"):
                series = group[metric_col].dropna()
                if not series.empty:
                    mode_series = series.mode()
                    mode_val = (
                        round(mode_series.iloc[0], 2)
                        if not mode_series.empty
                        else None
                    )
                    q75, q25 = stats.scoreatpercentile(series, [75, 25])

                    stats_list.append({
                        "Scenario": sc_name,
                        "Count (N)": int(series.count()),
                        "Mean": round(series.mean(), 2),
                        "Median": round(series.median(), 2),
                        "Mode": mode_val,
                        "Min": round(series.min(), 2),
                        "Max": round(series.max(), 2),
                        "Std. Dev": round(series.std(), 2),
                        "IQR": round(q75 - q25, 2),
                    })
            return pd.DataFrame(stats_list)

        st.subheader("📊 Performance Expectation Score Summary")
        st.dataframe(
            get_enhanced_stats(df_clean, "Expectation_Score"),
            use_container_width=True,
            hide_index=True,
        )

    # ------------------ Expander 2: Boxplots & Distribution ------------------
    with st.expander(
        "📦 Distribution Boxplot & Individual Data Points", expanded=False
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.set_theme(style="whitegrid")

        sns.boxplot(
            data=df_clean,
            x="scenario_type",
            y="Expectation_Score",
            ax=ax,
            palette="Set2",
            width=0.4,
            boxprops=dict(alpha=0.7),
        )
        sns.stripplot(
            data=df_clean,
            x="scenario_type",
            y="Expectation_Score",
            ax=ax,
            color="black",
            alpha=0.5,
            jitter=0.2,
            size=6,
        )
        ax.set_title(
            "Auditor Performance Rating by Outcome Condition", fontsize=12
        )
        ax.set_xlabel("Outcome Scenario Condition", fontsize=10)
        ax.set_ylabel(
            "Expectation Score (-5: Below, 0: Met, +5: Above)", fontsize=10
        )
        ax.set_ylim(-5.5, 5.5)

        plt.tight_layout()
        st.pyplot(fig)

    # ------------------ Expander 3: Hypothesis Testing (Independent t-test) ------------------
    with st.expander(
        "🧪 Independent Samples t-test (Comparing Means Between 2 Conditions)",
        expanded=False,
    ):
        no_mis_grp = df_clean[
            df_clean["scenario_type"] == "No Misstatement"
        ]["Expectation_Score"].dropna()
        mis_grp = df_clean[df_clean["scenario_type"] == "Misstatement"][
            "Expectation_Score"
        ].dropna()

        if not no_mis_grp.empty and not mis_grp.empty:
            # Independent 2-sample t-test (Welch's t-test assuming unequal variance)
            t_stat, p_val = stats.ttest_ind(
                no_mis_grp, mis_grp, equal_var=False
            )

            ttest_df = pd.DataFrame({
                "Metric": ["Expectation Score (-5 to +5)"],
                "No Misstatement Mean": [round(no_mis_grp.mean(), 2)],
                "Misstatement Mean": [round(mis_grp.mean(), 2)],
                "t-Statistic": [round(t_stat, 3)],
                "p-Value": [round(p_val, 4)],
                "Significance (p < 0.05)": [
                    "Yes 🟢" if p_val < 0.05 else "No 🔴"
                ],
            })

            st.dataframe(ttest_df, use_container_width=True, hide_index=True)
        else:
            st.warning(
                "Insufficient data across both scenarios to perform the t-test."
            )

else:
    st.info("ℹ️ Please run the simulation above to view complete analysis.")