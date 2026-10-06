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
# 1. Page & API Setup
# ----------------------------------------------------
st.set_page_config(
    page_title="Audit Decision-Making Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Auditor Decision-Making & KAM Disclosure Simulation")

# Retrieve API Key
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "⚠️ API Key is missing! Please configure 'GROQ_API_KEY' in Streamlit secrets."
    )
    st.stop()

client = Groq(api_key=api_key)

# ----------------------------------------------------
# 2. Sidebar Configuration
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
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "allam-2-7b",
    ],
)

csv_path = st.sidebar.text_input(
    "Output CSV File Name:", value="accountability_results.csv"
)

# Load existing CSV into session_state if available
if "df_data" not in st.session_state and os.path.exists(csv_path):
    st.session_state["df_data"] = pd.read_csv(csv_path)

# ----------------------------------------------------
# 3. Experimental Scenarios
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
# 4. Simulation Execution
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

                IMPORTANT: Output ONLY two numbers separated by a comma (e.g., "7, 85"). Do not include any extra text.
                """

                status_text.text(
                    f"Running {persona_id} - Scenario: {sc_name}..."
                )

                rev_score, bel_score = None, None

                # Retry logic to handle Rate Limit (429) errors
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

                        break  # Exit retry loop if successful

                    except Exception as err:
                        if "429" in str(err) and attempt < max_retries - 1:
                            time.sleep(3)  # Wait 3 seconds before retrying
                        else:
                            st.warning(
                                f"Error fetching response for {persona_id}: {err}"
                            )
                            break

                # Delay to prevent hitting rate limits
                time.sleep(0.5)

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

    except Exception as e:
        st.error(f"Execution Error: {e}")

# Display current stored data and Download Button outside the button click trigger
if "df_data" in st.session_state and not st.session_state["df_data"].empty:
    st.subheader("📋 Current Dataset Preview")
    st.dataframe(st.session_state["df_data"], use_container_width=True)

    csv_bytes = st.session_state["df_data"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV Results",
        data=csv_bytes,
        file_name=csv_path,
        mime="text/csv",
    )
