
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# STEP A: Set up the application
st.set_page_config(
    page_title="Malaysia Scientific Visualisation",
    layout="wide"
)

st.title("Malaysia: Live Births, Education and Household Income")
st.write("JIE42403 Scientific Visualisation | Malaysian State Data")


# STEP B: Load the three datasets
@st.cache_data
def load_data():
    births = pd.read_csv("birth_state.csv")
    school = pd.read_csv("completion_school_state.csv")
    income = pd.read_csv("hh_income_state.csv")

    for df in (births, school, income):
        df["year"] = pd.to_datetime(df["date"]).dt.year

    return births, school, income


births, school, income = load_data()

# Remove Malaysia total and use both sexes
school = school[
    (school["state"] != "Malaysia") &
    (school["sex"] == "both")
]


# STEP C: Create the interactive filters
common_years = sorted(
    set(births.year) & set(school.year) & set(income.year)
)

st.sidebar.header("Choose Your Data")

year = st.sidebar.selectbox(
    "Year",
    common_years,
    index=len(common_years) - 1
)

stage_names = {
    "primary": "Primary",
    "secondary_lower": "Lower secondary",
    "secondary_upper": "Upper secondary"
}

stage = st.sidebar.selectbox(
    "School Level",
    list(stage_names),
    index=2,
    format_func=lambda x: stage_names[x]
)

income_names = {
    "income_median": "Median income",
    "income_mean": "Mean income"
}

income_col = st.sidebar.selectbox(
    "Monthly Household Income",
    list(income_names),
    format_func=lambda x: income_names[x]
)

birth_names = {
    "rate": "Birth rate (per 1,000 people)",
    "abs": "Annual live births (count)"
}

birth_col = st.sidebar.selectbox(
    "Live Births Measure",
    list(birth_names),
    format_func=lambda x: birth_names[x]
)


# STEP D: Match data by state and year
selected_school = school[
    school["stage"] == stage
][["state", "year", "completion"]]

matched = births.merge(
    income, on=["state", "year"]
).merge(
    selected_school, on=["state", "year"]
)

current = matched[
    matched["year"] == year
].dropna(
    subset=[income_col, birth_col, "completion"]
).copy()

if current.empty:
    st.error("No matched data for these filters.")
    st.stop()


# STEP E: Describe correlation strength
def explain(r):
    if pd.isna(r):
        return "insufficient data"

    size = abs(r)

    if size < 0.10:
        degree = "negligible"
    elif size < 0.30:
        degree = "weak"
    elif size < 0.70:
        degree = "moderate"
    else:
        degree = "strong"

    direction = (
        "positive" if r > 0
        else "negative" if r < 0
        else "no"
    )

    return f"{degree} {direction} relationship"


# STEP F: Create three objective tabs
t1, t2, t3 = st.tabs([
    "1. Births & Income",
    "2. Education & Income",
    "3. Births & Education"
])


# OBJECTIVE 1: BAR CHART
with t1:
    st.subheader(
        "Objective 1: Annual Live Births and Household Income"
    )

    st.write(
        "To identify the relationship between annual "
        "live births and household income."
    )

    r = current[income_col].corr(current[birth_col])

    c1, c2 = st.columns(2)
    c1.metric("States Compared", len(current))
    c2.metric("Correlation (r)", f"{r:+.2f}")

    current["Income group"] = pd.qcut(
        current[income_col],
        4,
        labels=[
            "Lowest", "Lower-middle",
            "Upper-middle", "Highest"
        ]
    )

    grouped = current.groupby(
        "Income group", observed=True
    )[birth_col].mean().reset_index()

    fig = px.bar(
        grouped,
        x="Income group",
        y=birth_col,
        text_auto=".1f",
        title=f"Average Births by Income Group ({year})",
        labels={birth_col: birth_names[birth_col]}
    )

    fig.update_traces(
        marker_color="#24866C",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"**Interpretation:** In {year}, the state-level "
        f"relationship is {explain(r)} (r = {r:+.2f}). "
        "The bars show average values for four income groups."
    )


# OBJECTIVE 2: SCATTER PLOT
with t2:
    st.subheader(
        "Objective 2: School Completion and Household Income"
    )

    st.write(
        "To examine the relationship between school "
        "completion rates and household income."
    )

    r = current[income_col].corr(current["completion"])

    c1, c2 = st.columns(2)

    c1.metric(
        "Average Completion Rate",
        f"{current['completion'].mean():.1f}%"
    )
    c2.metric("Correlation (r)", f"{r:+.2f}")

    fig = px.scatter(
        current,
        x=income_col,
        y="completion",
        hover_name="state",
        title=f"Income vs School Completion ({year})",
        labels={
            income_col: "Monthly Household Income (RM)",
            "completion": "School Completion Rate (%)"
        }
    )

    fig.update_traces(
        marker=dict(color="#24866C", size=12)
    )

    # Add a trend line
    x = current[income_col].to_numpy()
    y = current["completion"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)

    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()],
        y=[
            slope * x.min() + intercept,
            slope * x.max() + intercept
        ],
        mode="lines",
        name="Trend Line",
        line=dict(color="#D0793D", dash="dash")
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"**Interpretation:** In {year}, the relationship "
        f"is {explain(r)} (r = {r:+.2f}). "
        "Each dot represents one Malaysian state."
    )


# OBJECTIVE 3: LINE CHART
with t3:
    st.subheader(
        "Objective 3: Annual Live Births and School Completion"
    )

    st.write(
        "To analyse the relationship between annual "
        "live births and school completion rates."
    )

    two_data = births.merge(
        selected_school,
        on=["state", "year"]
    )

    trend = pd.DataFrame([
        {
            "year": int(y),
            "r": df[birth_col].corr(df["completion"])
        }
        for y, df in two_data.groupby("year")
    ])

    current_r = trend.loc[
        trend.year == year, "r"
    ].iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("Years Compared", len(trend))
    c2.metric(
        f"{year} Correlation (r)",
        f"{current_r:+.2f}"
    )

    fig = px.line(
        trend,
        x="year",
        y="r",
        markers=True,
        title="Births vs School Completion (2016–2025)",
        labels={
            "year": "Year",
            "r": "Pearson Correlation (r)"
        }
    )

    fig.update_traces(
        line_color="#24866C",
        line_width=3,
        marker_size=8
    )

    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[-1, 1])
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"**Interpretation:** The relationship in {year} "
        f"is {explain(current_r)} (r = {current_r:+.2f}). "
        "Each point shows the correlation across states "
        "for that year. The line shows how it changes over time."
    )


# STEP G: Show data information
st.caption(
    "Source: OpenDOSM. Correlation does not prove causation. "
    "Household income is available for selected survey years."
)

with st.expander("View Matched Dataset"):
    st.dataframe(current, use_container_width=True)
