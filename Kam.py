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
# 1. Page Configuration & Custom Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="Audit Decision-Making & KAM Simulation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS اختصاصی برای زیباتر کردن کارت‌ها، دکمه‌ها و جداول
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">⚖️ Auditor Decision-Making & KAM Disclosure Simulation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Simulating auditor judgments under different Key Audit Matter (KAM) reporting conditions.</div>',
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
    st.image(
        "https://img.icons8.com/color/96/000000/combo-chart.png", width=80
    )
    st.header("⚙️ Configuration Settings")

    num_personas = st.number_input(
        "Number of Replications (Personas/Scenario):",
        min_value=1,
        max_value=500,
        value=25,
        step=1,
        help="Total runs per scenario type.",
    )

    model_name = st.selectbox(
        "Select LLM Model:",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
        ],
        help="Select an active LLM available in your environment.",
    )

    csv_path = st.text_input(
        "Output CSV File Name:", value="accountability_results.csv"
    )

    st.markdown("---")
    st.markdown("💡 **Tip:** Data persists in memory until re-simulated.")

# Initialize or Load Cached Data
if "df_data" not in st.session_state and os.path.exists(csv_path):
    try:
        st.session_state["df_data"] = pd.read_csv(csv_path)
    except Exception:
        pass

# ----------------------------------------------------
# 4. Experimental Scenarios (Tabs)
# ----------------------------------------------------
st.header("📝 1. Experimental Setup")

tab1, tab2 = st.tabs(["📋 Base Scenario Context", "🔬 Treatment Scenarios"])

with tab1:
    base_scenario_text = st.text_area(
        "Edit Base Scenario Context:",
        value="""Your client, ABC Integrated Products, Ltd., is a publicly traded manufacturing company headquartered in Melbourne, Australia. ABC Integrated is profitable and has experienced stable financial growth over the past five years. Its financial indicators, including liquidity and leverage, align with industry averages. Prior audits found no identifiable material weaknesses in the company’s internal controls.
Under company guidelines, Overall financial statement materiality is set at $1,000,000 based on net income, and Performance Materiality is set at $500,000. During the audit, we agreed this materiality level was appropriate. All standard audit tests have been completed by competent members of your audit team, and you are satisfied with the results. Aside from the unresolved matter described on the following page, we are not considering any other financial statement adjustments. We identified no significant qualitative materiality factors during this year’s audit.
The client believes the financial statements are fairly presented and insists on receiving an unqualified opinion as soon as possible. The client firmly opposes any proposed audit adjustments and is pressuring you to waive them all.
Because of product innovation and revisions, the client identified manufacturing equipment that may be impaired at the end of the reporting period. Under IAS 36 Impairment of Assets, the client estimated the equipment's recoverable amount. The client applied IFRS 13 Fair Value Measurement to determine fair value. As relevant observable inputs—such as quoted prices in an active market for this or similar equipment—were unavailable, the client used unobservable inputs, which are categorised as Level 3 inputs under the IFRS 13 fair value hierarchy. The Chief Financial Officer, David Vance, developed the unobservable inputs and valued the equipment using a discounted cash flow (DCF) model. The equipment’s recoverable amount was estimated at $3 million to $4 million, while its recorded value was $3,450,000. The CFO formally concluded that no impairment was required.
The audit team engaged the firm’s valuation specialists to assess the client’s estimate. The specialists provided the following advice: “We measure these assets based on discounted future cash flows, as there is no active market for these assets. Our estimated range for these assets is approximately $2,250,000 to $2,800,000. This range was developed using level 3 inputs under IFRS 13.”""",
        height=200,
    )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        nokam_text = st.text_area(
            "Scenario 1: Nokam (No KAM Disclosure)",
            value="In the audit environment, An independent auditor’s report contains only the auditor’s opinion and the basis for that opinion.",
            height=120,
        )
    with col2:
        kam_text = st.text_area(
            "Scenario 2: Kam (KAM Disclosure Required)",
            value="Auditing Standard ISA 701, Communicating Key Audit Matters in the Independent Auditor’s Report, requires auditors to disclose the Key Audit Matters. Thus the auditor has to include a paragraph for asset impairment and the way he dealt with it in auditing.",
            height=120,
        )

scenarios_dict = {"Nokam": nokam_text, "Kam": kam_text}

# ----------------------------------------------------
# 5. Simulation Execution
# ----------------------------------------------------
st.divider()
st.header("🚀 2. Run Simulation")

c1, c2, c3 = st.columns([1, 1, 2])
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
            label="📥 Download CSV Dataset",
            data=csv_bytes,
            file_name=csv_path,
            mime="text/csv",
            use_container_width=True,
        )

if run_btn:
    try:
        results = []
        global_persona_id = 1

        positions = ["Senior auditor", "Manager"]
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
                    f"Always respond strictly with numeric values as requested."
                )

                user_instruction = f"""
                Background: {base_scenario_text}
                Scenario: {sc_content}

                Please evaluate and provide two numerical scores:
                1. Revision score (from 1 to 10): How likely are you to make management adjust the fair value estimates?
                2. Believability score (from 0 to 100): How confident are you in your decision regarding the impairment?

                IMPORTANT: Output ONLY two numbers separated by a comma (e.g., "7, 85"). Do not include any extra text.
                """

                status_text.text(
                    f"🔄 Processing {persona_id} | Scenario: {sc_name}..."
                )

                rev_score, bel_score = None, None
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
                        numbers = re.findall(r"\d+(?:\.\d+)?", response_text)

                        if len(numbers) >= 2:
                            rev_score = float(numbers[0])
                            bel_score = float(numbers[1])
                        elif len(numbers) == 1:
                            rev_score = float(numbers[0])

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
                    "Revision_Score": rev_score,
                    "Believability_Score": bel_score,
                })

                completed += 1
                progress_bar.progress(completed / total_tasks)

        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_path, index=False)
        st.session_state["df_data"] = df_res

        status_text.empty()
        st.success(f"✅ Simulation completed! Generated {len(df_res)} records.")

    except Exception as e:
        st.error(f"Execution Error: {e}")

# ----------------------------------------------------
# 6. Current Dataset Display
# ----------------------------------------------------
if "df_data" in st.session_state and not st.session_state["df_data"].empty:
    with st.expander("🔍 View Raw Simulation Dataset", expanded=False):
        st.dataframe(st.session_state["df_data"], use_container_width=True)

# ----------------------------------------------------
# 7. Descriptive Statistics Section
# ----------------------------------------------------
st.divider()
st.header("📈 3. Descriptive Statistics Analysis")

if "df_data" in st.session_state and not st.session_state["df_data"].empty:
    df = st.session_state["df_data"].copy()

    df["Revision_Score"] = pd.to_numeric(df["Revision_Score"], errors="coerce")
    df["Believability_Score"] = pd.to_numeric(
        df["Believability_Score"], errors="coerce"
    )

    df_clean = df.dropna(subset=["Revision_Score", "Believability_Score"])

    if df_clean.empty:
        st.warning(
            "⚠️ Dataset has missing numeric scores. Try running simulation again."
        )
    else:

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

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Revision Score Summary")
            rev_stats = get_enhanced_stats(df_clean, "Revision_Score")
            st.dataframe(
                rev_stats, use_container_width=True, hide_index=True
            )

        with col2:
            st.subheader("🎯 Believability Score Summary")
            bel_stats = get_enhanced_stats(df_clean, "Believability_Score")
            st.dataframe(
                bel_stats, use_container_width=True, hide_index=True
            )

else:
    st.info("ℹ️ Please run the simulation above to view descriptive stats.")
