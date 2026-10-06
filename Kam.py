import csv
import io
import os
import random
import re
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import streamlit as st
from groq import Groq

# ----------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Audit Decision-Making Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Auditor Decision-Making & KAM Disclosure Simulation")
st.markdown(
    "This dashboard simulates auditor decision-making across **Key Audit Matters (KAM)** scenarios and performs empirical statistical analysis."
)

# ----------------------------------------------------
# 2. API Key Retrieval (Fully Automated & Hidden)
# ----------------------------------------------------
# Retrieve API Key securely from Streamlit Secrets or Environment Variables
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "⚠️ API Key is missing! Please configure 'GROQ_API_KEY' in Streamlit secrets."
    )
    st.stop()

# Initialize Groq Client automatically
client = Groq(api_key=api_key)

# ----------------------------------------------------
# 3. Sidebar Configuration
# ----------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")

num_personas = st.sidebar.number_input(
    "Number of Replications (Personas per Scenario):",
    min_value=1,
    max_value=500,
    value=25,
    step=1,
)

model_name = st.sidebar.selectbox(
    "Select Model:",
    ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
)

csv_path = st.sidebar.text_input(
    "Output CSV File Name:", value="accountability_results.csv"
)

# ----------------------------------------------------
# 4. Experimental Scenarios Display
# ----------------------------------------------------
st.header("📝 1. Experimental Scenarios")

col_base, col_scenarios = st.columns([1.2, 1])

with col_base:
    st.subheader("Base Scenario")
    base_scenario_text = st.text_area(
        "Base Scenario Context:",
        value="""Your client, ABC Integrated Products, Ltd., is a publicly traded manufacturing company headquartered in Melbourne, Australia. ABC Integrated is profitable and has experienced stable financial growth over the past five years. Its financial indicators, including liquidity and leverage, align with industry averages. Prior audits found no identifiable material weaknesses in the company’s internal controls.
Under company guidelines, Overall financial statement materiality is set at $1,000,000 based on net income, and Performance Materiality is set at $500,000. During the audit, we agreed this materiality level was appropriate. All standard audit tests have been completed by competent members of your audit team, and you are satisfied with the results. Aside from the unresolved matter described on the following page, we are not considering any other financial statement adjustments. We identified no significant qualitative materiality factors during this year’s audit.
The client believes the financial statements are fairly presented and insists on receiving an unqualified opinion as soon as possible. The client firmly opposes any proposed audit adjustments and is pressuring you to waive them all.
Because of product innovation and revisions, the client identified manufacturing equipment that may be impaired at the end of the reporting period. Under IAS 36 Impairment of Assets, the client estimated the equipment's recoverable amount. The client applied IFRS 13 Fair Value Measurement to determine fair value. As relevant observable inputs—such as quoted prices in an active market for this or similar equipment—were unavailable, the client used unobservable inputs, which are categorised as Level 3 inputs under the IFRS 13 fair value hierarchy. The Chief Financial Officer, David Vance, developed the unobservable inputs and valued the equipment using a discounted cash flow (DCF) model. The equipment’s recoverable amount was estimated at $3 million to $4 million, while its recorded value was $3,450,000. The CFO formally concluded that no impairment was required.
The audit team engaged the firm’s valuation specialists to assess the client’s estimate. The specialists provided the following advice: “We measure these assets based on discounted future cash flows, as there is no active market for these assets. Our estimated range for these assets is approximately $2,250,000 to $2,800,000. This range was developed using level 3 inputs under IFRS 13.”""",
        height=280,
    )

with col_scenarios:
    st.subheader("Treatment Scenarios")
    nokam_text = st.text_area(
        "Scenario 1: Nokam (No KAM Disclosure):",
        value="In the audit environment, An independent auditor’s report contains only the auditor’s opinion and the basis for that opinion.",
        height=110,
    )
    kam_text = st.text_area(
        "Scenario 2: Kam (KAM Disclosure Required):",
        value="Auditing Standard ISA 701, Communicating Key Audit Matters in the Independent Auditor’s Report, requires auditors to disclose the Key Audit Matters. Thus the auditor has to include a paragraph for asset impairment and the way he dealt with it in auditing.",
        height=110,
    )

scenarios_dict = {"Nokam": nokam_text, "Kam": kam_text}

# ----------------------------------------------------
# ----------------------------------------------------
# Simulation Execution with Robust Score Extraction
# ----------------------------------------------------
st.divider()
st.header("🚀 2. Run Simulation")

if st.button("Start Simulation"):
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

                IMPORTANT: Output ONLY the two numbers separated by a comma (e.g., "7, 85"). Do not include any extra text.
                """

                status_text.text(
                    f"Running {persona_id} - Scenario: {sc_name}..."
                )

                rev_score, bel_score = None, None

                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_instruction},
                        ],
                        temperature=0.3,  # Lower temperature for more consistent structural response
                    )

                    response_text = completion.choices[0].message.content.strip()

                    # Extract all numbers/floats from the model response
                    numbers = re.findall(r"\d+(?:\.\d+)?", response_text)

                    if len(numbers) >= 2:
                        rev_score = float(numbers[0])
                        bel_score = float(numbers[1])
                    elif len(numbers) == 1:
                        rev_score = float(numbers[0])

                except Exception as err:
                    st.warning(f"Error fetching response for {persona_id}: {err}")

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

        st.success(
            f"Simulation completed! {len(df_res)} records generated and saved."
        )
        st.dataframe(df_res)

    except Exception as e:
        st.error(f"Execution Error: {e}")
# ----------------------------------------------------
# 3. Descriptive Statistics
# ----------------------------------------------------
st.divider()
st.header("📈 3. Descriptive Statistics")

if st.button("Display Descriptive Statistics"):
    if "df_data" in st.session_state:
        df = st.session_state["df_data"].copy()

        # 1. Convert columns to numeric to avoid data-type issues
        df["Revision_Score"] = pd.to_numeric(
            df["Revision_Score"], errors="coerce"
        )
        df["Believability_Score"] = pd.to_numeric(
            df["Believability_Score"], errors="coerce"
        )

        # Drop NaN values for accurate statistical calculation
        df_clean = df.dropna(subset=["Revision_Score", "Believability_Score"])

        if df_clean.empty:
            st.error(
                "No valid numeric data found. Please run the simulation first!"
            )
        else:

            def get_enhanced_stats(data, metric_col):
                stats_list = []
                for sc_name, group in data.groupby("scenario_type"):
                    series = group[metric_col].dropna()

                    if not series.empty:
                        # Calculate Mode safely using pandas
                        mode_series = series.mode()
                        mode_val = (
                            round(mode_series.iloc[0], 2)
                            if not mode_series.empty
                            else None
                        )

                        # Calculate IQR (Q3 - Q1)
                        q75, q25 = stats.scoreatpercentile(series, [75, 25])
                        iqr_val = q75 - q25

                        stats_list.append({
                            "Scenario": sc_name,
                            "Count (N)": int(series.count()),
                            "Mean": round(series.mean(), 2),
                            "Median": round(series.median(), 2),
                            "Mode": mode_val,
                            "Min": round(series.min(), 2),
                            "Max": round(series.max(), 2),
                            "Std. Dev": round(series.std(), 2),
                            "IQR": round(iqr_val, 2),
                        })
                return pd.DataFrame(stats_list)

            # Display Tables
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📌 Revision Score Analysis")
                rev_stats = get_enhanced_stats(df_clean, "Revision_Score")
                st.dataframe(rev_stats, use_container_width=True)

            with col2:
                st.subheader("📌 Believability Score Analysis")
                bel_stats = get_enhanced_stats(df_clean, "Believability_Score")
                st.dataframe(bel_stats, use_container_width=True)

    else:
        st.warning(
            "No data found. Please run the simulation first or check CSV file."
        )

# ----------------------------------------------------
# 7. Hypothesis Testing (Independent t-Test)
# ----------------------------------------------------
st.divider()
st.header("🔬 4. Hypothesis Testing (Independent t-Test)")

if st.button("Compare Means (Mean Difference)"):
    if "df_data" in st.session_state:
        df = st.session_state["df_data"].dropna(
            subset=["Revision_Score", "Believability_Score"]
        )

        nokam_group = df[df["scenario_type"] == "Nokam"]
        kam_group = df[df["scenario_type"] == "Kam"]

        if len(nokam_group) == 0 or len(kam_group) == 0:
            st.error("Both 'Nokam' and 'Kam' scenarios are required for t-test.")
        else:
            for metric in ["Revision_Score", "Believability_Score"]:
                st.subheader(f"Independent Samples t-Test for: {metric}")

                group_nokam = nokam_group[metric]
                group_kam = kam_group[metric]

                t_stat, p_val = stats.ttest_ind(group_nokam, group_kam)

                mean_nokam = group_nokam.mean()
                mean_kam = group_kam.mean()
                diff = mean_kam - mean_nokam

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Mean (Nokam)", f"{mean_nokam:.2f}")
                col_res2.metric("Mean (Kam)", f"{mean_kam:.2f}")
                col_res3.metric("Mean Difference (Kam - Nokam)", f"{diff:.2f}")

                st.write(f"**t-statistic:** `{t_stat:.4f}`")
                st.write(f"**p-value:** `{p_val:.4f}`")

                if p_val < 0.05:
                    st.success(
                        "✅ **Statistically Significant Difference** detected between scenarios ($p < 0.05$)."
                    )
                else:
                    st.info(
                        "ℹ️ **No Statistically Significant Difference** detected between scenarios ($p \ge 0.05$)."
                    )
                st.markdown("---")
    else:
        st.warning("Please execute the simulation first.")

# ----------------------------------------------------
# 8. Boxplot Visualization
# ----------------------------------------------------
st.divider()
st.header("📦 5. Visualization (Boxplots)")

if st.button("Generate Boxplots"):
    if "df_data" in st.session_state:
        df = st.session_state["df_data"].dropna(
            subset=["Revision_Score", "Believability_Score"]
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.set_theme(style="whitegrid")

        # Revision Score Boxplot
        sns.boxplot(
            ax=axes[0],
            data=df,
            x="scenario_type",
            y="Revision_Score",
            palette="Blues",
        )
        axes[0].set_title(
            "Revision Score Comparison", fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel("Scenario", fontsize=10)
        axes[0].set_ylabel("Score (1 - 10)", fontsize=10)

        # Believability Score Boxplot
        sns.boxplot(
            ax=axes[1],
            data=df,
            x="scenario_type",
            y="Believability_Score",
            palette="Greens",
        )
        axes[1].set_title(
            "Believability Score Comparison", fontsize=12, fontweight="bold"
        )
        axes[1].set_xlabel("Scenario", fontsize=10)
        axes[1].set_ylabel("Score (0 - 100%)", fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Please execute the simulation first.")
