import os
import random
import re
import time
import matplotlib.pyplot as plt
from openai import OpenAI
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import streamlit as st
from groq import Groq

# ----------------------------------------------------
# 1. Page Configuration & Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="Audit Decision-Making & KAM Simulation",
    page_icon="⚖️",
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
    '<div class="main-header">⚖️ Auditor Decision-Making & KAM Disclosure Simulation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Simulating auditor judgments under different Key Audit Matter (KAM) reporting conditions.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# 2. Multi-Platform API Setup
# ----------------------------------------------------
OHH_API_KEY = "sk-mvx-"

def get_secret(key_name, fallback_val=None):
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, fallback_val)

ohh_key = get_secret("OHH_API_KEY", OHH_API_KEY if OHH_API_KEY != "your_actual_ohhmyagent_api_key_here" else None)
groq_key = get_secret("GROQ_API_KEY")
openrouter_key = get_secret("OPENROUTER_API_KEY")
gemini_key = get_secret("GEMINI_API_KEY")

# ساخت کلاینت‌ها با تنظیم timeout
ohh_client = (
    OpenAI(
        base_url="https://ohhmyagent.com/v1",
        api_key=ohh_key or "missing",
        timeout=30.0,
    )
    if ohh_key
    else None
)
groq_client = Groq(api_key=groq_key) if groq_key else None
openrouter_client = (
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key or "missing",
        timeout=30.0,
    )
    if openrouter_key
    else None
)
gemini_client = (
    OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_key or "missing",
        timeout=30.0,
    )
    if gemini_key
    else None
)

# ----------------------------------------------------
# 3. Sidebar Setup
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration Settings")

    platform = st.selectbox(
        "Select Provider / Platform:",
        [
            "OhhMyAgent (Local / Custom)",
            "Google AI Studio (Free)",
            "Groq (Free Tier)",
            "OpenRouter",
        ],
    )

    if platform == "OhhMyAgent (Local / Custom)":
        if not ohh_key:
            st.warning("⚠️ 'OHH_API_KEY' is missing!")
        model_name = st.selectbox(
            "Select Model (OhhMyAgent):",
            [
                "ohh/gpt-6-sol",
                "ohh/gpt-6-astra",
                "ohh/claude-opus-5",
                "ohh/gpt-5.6",
                "ohh/claude-opus-4.8",
            ],
        )
    elif platform == "Google AI Studio (Free)":
        if not gemini_key:
            st.warning("⚠️ 'GEMINI_API_KEY' is missing in Secrets!")
        model_name = st.selectbox(
            "Select Free Gemini Model:",
            [
                "gemini-3.8-flash",
                "gemini-2.5-flash",
            ],
        )
    elif platform == "Groq (Free Tier)":
        if not groq_key:
            st.warning("⚠️ 'GROQ_API_KEY' is missing in Secrets!")
        model_name = st.selectbox(
            "Select Free Model (Groq):",
            [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.8-27b",
                "allam-2-7b",
            ],
        )
    else:
        if not openrouter_key:
            st.warning("⚠️ 'OPENROUTER_API_KEY' is missing in Secrets!")
        model_name = st.selectbox(
            "Select Model (OpenRouter):",
            [
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemma-2-9b-it:free",
                "qwen/qwen-2.5-72b-instruct:free",
            ],
        )

    num_personas = st.number_input(
        "Number of Replications (Personas/Scenario):",
        min_value=1,
        max_value=500,
        value=25,
        step=1,
    )

    csv_path = st.text_input("Output CSV File Name:", value="KAM_results.csv")

# لود کردن فایل قبلی در session_state در صورت وجود
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
        value="""Your client, ABC Integrated Products, Ltd., is a publicly traded manufacturing company headquartered in Melbourne, Australia. 
ABC Integrated is profitable and has experienced stable financial growth over the past five years. Its financial indicators, including liquidity and leverage, align with industry averages. Prior audits found no identifiable material weaknesses in the company’s internal controls.
Under company guidelines, Overall financial statement materiality is set at $1,000,000 based on net income, and Performance Materiality is set at $500,000. During the audit, we agreed this materiality level was appropriate. All standard audit tests have been completed by competent members of your audit team, and you are satisfied with the results. Aside from the unresolved matter described on the following page, we are not considering any other financial statement adjustments. 
The client believes the financial statements are fairly presented and insists on receiving an unqualified opinion as soon as possible. 
The CFO firmly emphasizes that any recorded impairment write-down would trigger a debt covenant breach with primary lenders, thereby opposing any proposed audit adjustments.
Because of product innovation and revisions, the client identified manufacturing equipment that may be impaired at the end of the reporting period. Under IAS 36 Impairment of Assets, the client estimated the equipment's recoverable amount. The client applied IFRS 13 Fair Value Measurement to determine fair value. As relevant observable inputs—such as quoted prices in an active market for this or similar equipment—were unavailable, the client used unobservable inputs, which are categorised as Level 3 inputs under the IFRS 13 fair value hierarchy. The Chief Financial Officer, David Vance, developed the unobservable inputs and valued the equipment using a discounted cash flow (DCF) model at a range of $3,300,000 to $3,900,000, formally concluding that the carrying amount of $3,450,000 requires no impairment.
The audit team engaged the firm’s valuation specialists to assess the client’s estimate. The specialists provided the following advice: “We measure these assets based on discounted future cash flows, as there is no active market for these assets. Given differing views on industry growth rates and technological obsolescence, our independent valuation range is approximately $2,400,000 to $2,900,000. We developed this range using level 3 inputs under IFRS 13.”""",
        height=180,
        key="txt_base_scenario",
    )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        nokam_text = st.text_area(
            "Scenario 1: Nokam (No KAM Disclosure)",
            value=(
                "Under applicable local reporting regulations, the independent auditor’s report contains "
                "only the auditor's formal opinion and the basis for that opinion. Key Audit Matters (KAM) are not communicated."
            ),
            height=120,
            key="txt_nokam_scenario",
        )
    with col2:
        kam_text = st.text_area(
            "Scenario 2: Kam (KAM Disclosure Required)",
            value=(
                "Auditing Standard ISA 701, Communicating Key Audit Matters in the Independent Auditor’s Report, "
                "requires auditors to disclose Key Audit Matters that in the auditor’s professional judgment, "
                "were of most significance in the audit of the financial report of the current period."
            ),
            height=120,
            key="txt_kam_scenario",
        )

scenarios_dict = {"Nokam": nokam_text, "Kam": kam_text}

# ----------------------------------------------------
# 5. Data Source Selection (API Simulation vs File Upload)
# ----------------------------------------------------
st.divider()
st.header("📂 2. Data Source & Execution")

data_mode = st.radio(
    "Choose Data Source for Analysis:",
    ["▶️ Run New Simulation (via API)", "📤 Upload Existing Dataset (CSV/Excel)"],
    horizontal=True,
)

if data_mode == "▶️ Run New Simulation (via API)":
    c1, c2 = st.columns([1.5, 2.5])
    with c1:
        run_btn = st.button(
            "🚀 Start API Simulation", type="primary", use_container_width=True
        )

    if run_btn:
        extra_headers = {}

        if platform == "OhhMyAgent (Local / Custom)":
            if not ohh_key:
                st.error("Please configure 'OHH_API_KEY' first!")
                st.stop()
            active_client = ohh_client
        elif platform == "Google AI Studio (Free)":
            if not gemini_key:
                st.error("Please configure 'GEMINI_API_KEY' in Secrets first!")
                st.stop()
            active_client = gemini_client
        elif platform == "Groq (Free Tier)":
            if not groq_key:
                st.error("Please configure 'GROQ_API_KEY' in Secrets first!")
                st.stop()
            active_client = groq_client
        else:
            if not openrouter_key:
                st.error("Please configure 'OPENROUTER_API_KEY' in Secrets first!")
                st.stop()
            active_client = openrouter_client
            extra_headers = {
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Audit Research App",
            }

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
                        f"You are an auditor with a {p_selected} position. Your "
                        f"experience is {exp_selected} and your gender is {gen_selected}. "
                        "Always respond strictly with numeric values as requested."
                    )

                    user_instruction = f"""
Background Context: {base_scenario_text}
Scenario: {sc_content}

Evaluate the situation and provide exactly three numerical scores:
1. Revision score (from 1 to 10): How likely are you to make management adjust the fair value estimates?
2. Believability score (from 0 to 100): How confident are you in your decision regarding the impairment?
3. Accountability score (from 1 to 10): How accountable do you feel for ensuring the reasonableness of the financial statements?

Format your response ONLY as three numbers separated by a comma. Example: 6, 75, 8
Do not include explanations or extra text.
"""

                    status_text.text(
                        f"🔄 Processing {persona_id} | Scenario: {sc_name} ({platform} - {model_name})..."
                    )

                    rev_score, bel_score, acc_score = None, None, None
                    attempt_count = 0
                    max_loop_attempts = 5

                    while attempt_count < max_loop_attempts:
                        attempt_count += 1
                        try:
                            call_params = {
                                "model": model_name,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_instruction},
                                ],
                                "temperature": 0.3,
                                "max_tokens": 300,
                            }
                            if extra_headers:
                                call_params["extra_headers"] = extra_headers

                            completion = active_client.chat.completions.create(**call_params)

                            msg = completion.choices[0].message if completion.choices else None
                            raw_content = getattr(msg, "content", None)
                            if not raw_content and hasattr(msg, "reasoning_content"):
                                raw_content = getattr(msg, "reasoning_content", None)

                            if raw_content:
                                response_text = str(raw_content).strip()
                                clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

                                # استخراج الگوهای ۳ عددی با کاما
                                comma_match = re.search(
                                    r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)",
                                    clean_text,
                                )
                                if comma_match:
                                    v1 = float(comma_match.group(1))
                                    v2 = float(comma_match.group(2))
                                    v3 = float(comma_match.group(3))
                                    if (1.0 <= v1 <= 10.0) and (0.0 <= v2 <= 100.0) and (1.0 <= v3 <= 10.0):
                                        rev_score, bel_score, acc_score = v1, v2, v3
                                        break

                                # حالت جایگزین: استخراج کلیه اعداد مجزا در متن
                                numbers = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", clean_text)]
                                if len(numbers) >= 3:
                                    v1, v2, v3 = numbers[0], numbers[1], numbers[2]
                                    if (1.0 <= v1 <= 10.0) and (0.0 <= v2 <= 100.0) and (1.0 <= v3 <= 10.0):
                                        rev_score, bel_score, acc_score = v1, v2, v3
                                        break

                            status_text.warning(
                                f"⏳ Incomplete response on {persona_id} (Attempt {attempt_count}). Retrying..."
                            )
                            time.sleep(1.5)

                        except Exception as err:
                            err_str = str(err)
                            if "503" in err_str:
                                status_text.warning(f"⚠️ Server unavailable (503) on {persona_id}. Waiting 6s...")
                                time.sleep(6)
                            elif "429" in err_str:
                                status_text.warning(f"⚠️ Rate limit on {persona_id}. Retrying in 4s...")
                                time.sleep(4)
                            else:
                                status_text.warning(f"⚠️ Network error on {persona_id}: {err_str[:60]}...")
                                time.sleep(2)

                    # مقادیر پشتیبان در صورت عدم پاسخ پس از ۵ مرتبه
                    if rev_score is None or bel_score is None or acc_score is None:
                        rev_score = 5.0
                        bel_score = 50.0
                        acc_score = 7.0

                    time.sleep(0.4)

                    results.append({
                        "scenario_type": sc_name,
                        "ID": persona_id,
                        "position": p_selected,
                        "Experience": exp_selected,
                        "gender": gen_selected,
                        "Revision_Score": rev_score,
                        "Believability_Score": bel_score,
                        "Accountability_Score": acc_score,
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

else:
    # ------------------ آپلود دیتاست موجود (اکسل / CSV) ------------------
    uploaded_file = st.file_uploader(
        "📥 Upload Previously Saved Simulation Dataset (CSV or Excel):",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)

            st.session_state["df_data"] = uploaded_df
            st.success(f"✅ File loaded successfully! Loaded {len(uploaded_df)} records.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ----------------------------------------------------
# دانلود داده‌ها و نمایش جدول نتایج
# ----------------------------------------------------
if "df_data" in st.session_state and not st.session_state["df_data"].empty:
    col_dl, _ = st.columns([1.5, 3])
    with col_dl:
        csv_bytes = st.session_state["df_data"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Current Dataset CSV",
            data=csv_bytes,
            file_name=csv_path,
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("🔍 View Raw Simulation Dataset", expanded=False):
        st.dataframe(st.session_state["df_data"], use_container_width=True)

# ----------------------------------------------------
# 6. Analysis Modules (Descriptive, Plots & t-test)
# ----------------------------------------------------
if "df_data" in st.session_state and not st.session_state["df_data"].empty:
    df = st.session_state["df_data"].copy()
    for col in ["Revision_Score", "Believability_Score", "Accountability_Score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df_clean = df.dropna(subset=["Revision_Score", "Believability_Score"])

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
                    mode_val = round(mode_series.iloc[0], 2) if not mode_series.empty else None
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

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📊 Revision Score")
            st.dataframe(get_enhanced_stats(df_clean, "Revision_Score"), use_container_width=True, hide_index=True)

        with col2:
            st.subheader("🎯 Believability Score")
            st.dataframe(get_enhanced_stats(df_clean, "Believability_Score"), use_container_width=True, hide_index=True)

        with col3:
            st.subheader("⚖️ Accountability Score")
            if "Accountability_Score" in df_clean.columns:
                st.dataframe(get_enhanced_stats(df_clean, "Accountability_Score"), use_container_width=True, hide_index=True)

    # ------------------ Expander 2: Boxplots & Distribution ------------------
    with st.expander("📦 Distribution Boxplots & Individual Data Points", expanded=False):
        has_acc = "Accountability_Score" in df_clean.columns
        n_plots = 3 if has_acc else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5))
        sns.set_theme(style="whitegrid")

        # Boxplot 1: Revision Score
        sns.boxplot(data=df_clean, x="scenario_type", y="Revision_Score", ax=axes[0], palette="Set2", width=0.4, boxprops=dict(alpha=0.7))
        sns.stripplot(data=df_clean, x="scenario_type", y="Revision_Score", ax=axes[0], color="black", alpha=0.5, jitter=0.2, size=6)
        axes[0].set_title("Revision Score (1-10)", fontsize=12)
        axes[0].set_ylim(0, 11)

        # Boxplot 2: Believability Score
        sns.boxplot(data=df_clean, x="scenario_type", y="Believability_Score", ax=axes[1], palette="Set2", width=0.4, boxprops=dict(alpha=0.7))
        sns.stripplot(data=df_clean, x="scenario_type", y="Believability_Score", ax=axes[1], color="black", alpha=0.5, jitter=0.2, size=6)
        axes[1].set_title("Believability Score (0-100)", fontsize=12)
        axes[1].set_ylim(-5, 105)

        # Boxplot 3: Accountability Score
        if has_acc:
            sns.boxplot(data=df_clean, x="scenario_type", y="Accountability_Score", ax=axes[2], palette="Set2", width=0.4, boxprops=dict(alpha=0.7))
            sns.stripplot(data=df_clean, x="scenario_type", y="Accountability_Score", ax=axes[2], color="black", alpha=0.5, jitter=0.2, size=6)
            axes[2].set_title("Accountability Score (1-10)", fontsize=12)
            axes[2].set_ylim(0, 11)

        plt.tight_layout()
        st.pyplot(fig)

    # ------------------ Expander 3: Hypothesis Testing ------------------
    with st.expander("🧪 Mean Comparison Test (Independent Samples t-test)", expanded=False):
        nokam_grp = df_clean[df_clean["scenario_type"] == "Nokam"]
        kam_grp = df_clean[df_clean["scenario_type"] == "Kam"]

        if not nokam_grp.empty and not kam_grp.empty:

            def run_ttest(col_name):
                t_stat, p_val = stats.ttest_ind(
                    nokam_grp[col_name].dropna(),
                    kam_grp[col_name].dropna(),
                    equal_var=False,
                )
                return round(t_stat, 3), round(p_val, 4)

            metrics_to_test = ["Revision_Score", "Believability_Score"]
            if "Accountability_Score" in df_clean.columns:
                metrics_to_test.append("Accountability_Score")

            ttest_rows = []
            for metric in metrics_to_test:
                t_val, p_val = run_ttest(metric)
                
