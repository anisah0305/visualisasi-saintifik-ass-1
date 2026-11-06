from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure
from bokeh.resources import CDN
from matplotlib.ticker import StrMethodFormatter


# ==========================================
# 1. DASHBOARD DESIGN
# ==========================================

st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide"
)

GREEN = "#298A6B"
DARK = "#174737"
ORANGE = "#D7854A"

sns.set_theme(style="whitegrid")

st.markdown("""
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
}

h1, h2, h3 {
    color: #174737;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATA LOADING AND CLEANING
# ==========================================

@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent

    birth = pd.read_csv(folder / "birth_state.csv")
    income = pd.read_csv(folder / "hh_income_state.csv")
    school = pd.read_csv(
        folder / "completion_school_state.csv"
    )

    # Standardize state names and dates
    for df in (birth, income, school):
        df["state"] = df["state"].astype(str).str.strip()
        df["year"] = pd.to_datetime(
            df["date"], errors="coerce"
        ).dt.year

    # Convert numerical columns
    for col in ("abs", "rate"):
        birth[col] = pd.to_numeric(
            birth[col], errors="coerce"
        )

    for col in ("income_median", "income_mean"):
        income[col] = pd.to_numeric(
            income[col], errors="coerce"
        )

    school["completion"] = pd.to_numeric(
        school["completion"], errors="coerce"
    )

    school["stage"] = school["stage"].str.strip()
    school["sex"] = school["sex"].str.strip()

    # Remove duplicate observations
    birth = birth.dropna(
        subset=["state", "year"]
    ).drop_duplicates(["state", "year"])

    income = income.dropna(
        subset=["state", "year"]
    ).drop_duplicates(["state", "year"])

    # Exclude Malaysia national summary
    school = school.loc[school["state"] != "Malaysia"]

    school = school.dropna(
        subset=["state", "year", "stage", "sex"]
    ).drop_duplicates(
        ["state", "year", "stage", "sex"]
    )

    # Merge datasets
    birth_income = birth.merge(
        income,
        on=["state", "year"],
        validate="one_to_one"
    )

    school_income = school.merge(
        income,
        on=["state", "year"],
        validate="many_to_one"
    )

    birth_school = birth.merge(
        school,
        on=["state", "year"],
        validate="one_to_many"
    )

    return birth_income, school_income, birth_school


try:
    bi, si, bs = load_data()

except Exception as error:
    st.error(f"Could not load the CSV files: {error}")
    st.stop()


# ==========================================
# 3. INTERACTIVE DASHBOARD FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

shared_years = sorted(
    set(bi["year"]) & set(si["year"])
)

year = st.sidebar.selectbox(
    "Household income survey year",
    shared_years,
    index=len(shared_years) - 1
)

stages = {
    "secondary_upper": "Upper secondary",
    "secondary_lower": "Lower secondary",
    "primary": "Primary"
}

stage = st.sidebar.selectbox(
    "Education stage",
    list(stages),
    format_func=stages.get
)

sexes = {
    "both": "Both genders",
    "female": "Female",
    "male": "Male"
}

sex = st.sidebar.selectbox(
    "Gender",
    list(sexes),
    format_func=sexes.get
)

income_labels = {
    "income_median": "Median income",
    "income_mean": "Mean income"
}

income_col = st.sidebar.selectbox(
    "Monthly income measure",
    list(income_labels),
    format_func=income_labels.get
)

st.sidebar.caption(
    "The selected survey year applies to Objectives 1 "
    "and 2. Objective 3 shows 2016–2025."
)


# ==========================================
# 4. DASHBOARD HEADER
# ==========================================

st.markdown("""
<div style="
background:#174737;
color:white;
padding:24px 28px;
border-radius:14px;
margin-bottom:18px;
">
    <div style="
    font-size:13px;
    color:#DDECE4;
    font-weight:600;
    ">
        JIE42403 · SCIENTIFIC VISUALISATION
    </div>

    <div style="
    font-size:32px;
    font-weight:750;
    ">
        Malaysia Data Insights
    </div>

    <div style="color:#E7F4EC">
        Birth Rates · Household Income · School Completion
    </div>
</div>
""", unsafe_allow_html=True)


st.subheader("Research Objectives")

with st.container(border=True):

    st.write(
        "**Objective 1:** To identify the relationship "
        "between annual live births and household income."
    )

    st.write(
        "**Objective 2:** To examine the relationship "
        "between school completion rates and household income."
    )

    st.write(
        "**Objective 3:** To analyse the relationship "
        "between annual live births and school completion rates."
    )

st.caption(
    "For comparisons between states, annual live births "
    "are represented as births per 1,000 people."
)


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def correlation(df, x, y):

    d = df[[x, y]].dropna()

    if (
        len(d) < 3
        or d[x].nunique() < 2
        or d[y].nunique() < 2
    ):
        return float("nan")

    return float(d[x].corr(d[y]))


def describe(r):

    if pd.isna(r):
        return "undefined"

    if abs(r) < 0.1:
        strength = "negligible"
    elif abs(r) < 0.3:
        strength = "weak"
    elif abs(r) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    if r > 0:
        direction = "positive"
    elif r < 0:
        direction = "negative"
    else:
        direction = "neither positive nor negative"

    return f"{strength} {direction}"


def cards(items):

    columns = st.columns(len(items))

    for col, (label, value) in zip(columns, items):
        with col:
            st.metric(label, value, border=True)


# ==========================================
# 6. OBJECTIVE 1 - SEABORN BAR CHART
# ==========================================

def objective1():

    st.header(
        "Objective 1 — Annual Live Births and Household Income"
    )

    st.write(
        "**Objective:** To identify the relationship "
        "between annual live births and household income."
    )

    a = bi.loc[
        bi["year"] == year,
        ["state", "year", "rate", "abs", income_col]
    ].dropna().copy()

    if len(a) < 4:
        st.warning("Not enough matching state records.")
        return

    r = correlation(a, income_col, "rate")

    # Summary boxes
    cards([
        ("Survey year", str(year)),
        ("States compared", str(a.state.nunique())),
        ("Pearson r", f"{r:+.3f}")
    ])

    # Divide states into four relative income groups
    a["Income group"] = pd.qcut(
        a[income_col].rank(method="first"),
        4,
        labels=[
            "Lowest",
            "Lower-middle",
            "Upper-middle",
            "Highest"
        ]
    )

    grouped = a.groupby(
        "Income group",
        observed=True
    )["rate"].mean().reset_index()

    # ONE VISUALIZATION
    st.subheader(
        "Average Birth Rate by Household Income Group"
    )

    fig, ax = plt.subplots(figsize=(9, 4.7))

    sns.barplot(
        data=grouped,
        x="Income group",
        y="rate",
        color=GREEN,
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            padding=4
        )

    ax.set_xlabel("Relative Household Income Group")
    ax.set_ylabel("Live Births per 1,000 People")

    ax.set_ylim(
        0,
        grouped["rate"].max() * 1.2
    )

    sns.despine(ax=ax)
    fig.tight_layout()

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Interpretation
    st.subheader("Interpretation and Discussion")

    st.write(
        f"In {year}, the correlation between household "
        f"income and birth rate was **{r:+.3f}**, "
        f"indicating a **{describe(r)}** state-level "
        "relationship. The four bars show average birth "
        "rates for equally sized relative income groups, "
        "not official income classifications. "
        "This association does not establish causation."
    )

    # Data preview
    with st.expander("Data Preview"):

        st.dataframe(
            a[["state", income_col, "rate", "abs"]],
            hide_index=True,
            use_container_width=True
        )


# ==========================================
# 7. OBJECTIVE 2 - SEABORN SCATTER PLOT
# ==========================================

def objective2():

    st.header(
        "Objective 2 — School Completion and Household Income"
    )

    st.write(
        "**Objective:** To examine the relationship "
        "between school completion rates and household income."
    )

    b = si.loc[
        (si["year"] == year)
        & (si["stage"] == stage)
        & (si["sex"] == sex),
        ["state", "year", "completion", income_col]
    ].dropna().copy()

    if len(b) < 3:
        st.warning("Not enough matching observations.")
        return

    r = correlation(
        b,
        income_col,
        "completion"
    )

    # Summary boxes
    cards([
        ("Survey year", str(year)),
        (
            "Average completion",
            f"{b.completion.mean():.1f}%"
        ),
        ("Pearson r", f"{r:+.3f}")
    ])

    # ONE VISUALIZATION
    st.subheader(
        "Household Income vs School Completion Rate"
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))

    sns.regplot(
        data=b,
        x=income_col,
        y="completion",
        ci=None,
        scatter_kws={
            "color": GREEN,
            "s": 85,
            "alpha": 0.85
        },
        line_kws={
            "color": ORANGE,
            "linewidth": 2.5
        },
        ax=ax
    )

    ax.set_xlabel("Monthly Household Income (RM)")
    ax.set_ylabel("School Completion Rate (%)")

    ax.xaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )

    sns.despine(ax=ax)
    fig.tight_layout()

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Interpretation
    st.subheader("Interpretation and Discussion")

    st.write(
        f"For **{stages[stage].lower()}** and "
        f"**{sexes[sex].lower()}** in {year}, the "
        f"correlation was **{r:+.3f}**, a "
        f"**{describe(r)}** relationship. "
        "Each dot represents one state or federal "
        "territory. The orange regression line indicates "
        "the overall linear direction, but does not "
        "prove that income causes completion rates "
        "to change."
    )

    # Data preview
    with st.expander("Data Preview"):

        st.dataframe(
            b,
            hide_index=True,
            use_container_width=True
        )


# ==========================================
# 8. OBJECTIVE 3 - BOKEH LINE CHART
# ==========================================

def objective3():

    st.header(
        "Objective 3 — Annual Live Births and School Completion"
    )

    st.write(
        "**Objective:** To analyse the relationship "
        "between annual live births and school completion rates."
    )

    c = bs.loc[
        (bs["stage"] == stage)
        & (bs["sex"] == sex)
    ].copy()

    # Calculate yearly Pearson correlations
    rows = []

    for yr, group in c.groupby("year"):

        rows.append({
            "Year": int(yr),
            "Correlation": correlation(
                group,
                "rate",
                "completion"
            ),
            "States": group["state"].nunique()
        })

    trend = pd.DataFrame(rows)

    trend = trend.sort_values("Year").dropna(
        subset=["Correlation"]
    )

    if trend.empty:
        st.warning("No valid yearly correlations found.")
        return

    latest = trend.iloc[-1]

    # Summary boxes
    cards([
        ("Years analysed", str(len(trend))),
        ("States covered", str(c.state.nunique())),
        (
            f"Correlation in {int(latest.Year)}",
            f"{latest.Correlation:+.3f}"
        )
    ])

    # ONE VISUALIZATION
    st.subheader(
        "Yearly Correlation: Birth Rates vs School Completion"
    )

    source = ColumnDataSource(trend)

    p = figure(
        x_axis_label="Year",
        y_axis_label="Pearson Correlation (r)",
        height=430,
        sizing_mode="stretch_width",
        x_range=(2015.5, 2025.5),
        y_range=(-1, 1),
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    # Draw line
    p.line(
        "Year",
        "Correlation",
        source=source,
        line_width=3,
        color=DARK
    )

    # Add points
    points = p.scatter(
        "Year",
        "Correlation",
        source=source,
        size=11,
        color=GREEN
    )

    # Hover tooltips
    p.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=[
                ("Year", "@Year"),
                (
                    "Pearson r",
                    "@Correlation{0.000}"
                ),
                ("States compared", "@States")
            ]
        )
    )

    # Reference line at zero
    p.add_layout(
        Span(
            location=0,
            dimension="width",
            line_color="gray",
            line_dash="dashed"
        )
    )

    p.xaxis.ticker = list(range(2016, 2026))
    p.ygrid.grid_line_alpha = 0.2
    p.toolbar.logo = None

    # Display interactive Bokeh chart
    st.iframe(
        file_html(
            p,
            CDN,
            "Births and School Completion"
        ),
        height=490,
        alt="Interactive line chart of yearly correlations"
    )

    # Identify highest and lowest correlations
    high = trend.loc[
        trend["Correlation"].idxmax()
    ]

    low = trend.loc[
        trend["Correlation"].idxmin()
    ]

    # Interpretation
    st.subheader("Interpretation and Discussion")

    st.write(
        "Across 2016–2025, correlations between birth "
        f"rates and **{stages[stage].lower()}** completion "
        "changed over time. The highest correlation was "
        f"**{high.Correlation:+.3f}** in **{int(high.Year)}**, "
        "and the lowest was "
        f"**{low.Correlation:+.3f}** in **{int(low.Year)}**. "
        "The latest correlation was "
        f"**{latest.Correlation:+.3f}**, indicating a "
        f"**{describe(latest.Correlation)}** relationship. "
        "These yearly state-level associations do not "
        "demonstrate causation."
    )

    # Data preview
    with st.expander("Data Preview"):

        st.dataframe(
            trend,
            hide_index=True,
            use_container_width=True
        )


# ==========================================
# 9. OBJECTIVE PAGE NAVIGATION
# ==========================================

navigation = st.navigation([

    st.Page(
        objective1,
        title="01 · Births & Income",
        icon="📊",
        default=True
    ),

    st.Page(
        objective2,
        title="02 · Education & Income",
        icon="🎓"
    ),

    st.Page(
        objective3,
        title="03 · Births & Education",
        icon="📈"
    )

], position="sidebar")

navigation.run()


# ==========================================
# 10. DATASET SOURCES AND METHODOLOGY
# ==========================================

st.divider()

with st.expander("Dataset Sources and Methodology"):

    st.write(
        "**Sources:** OpenDOSM state-level live births, "
        "household income, and school completion datasets."
    )

    st.write(
        "**Data preparation:** Dates were converted to "
        "years, numerical columns checked, duplicate "
        "observations removed, nationwide school totals "
        "excluded, and datasets merged by state and year."
    )

    st.write(
        "**Limitations:** Income surveys cover selected "
        "years only. School completion rates may exceed "
        "100%. The findings represent state-level "
        "associations rather than causal effects."
    )

    st.markdown(
        "[School Completion]"
        "(https://open.dosm.gov.my/data-catalogue/completion_school_state)"
        " · "
        "[Household Income]"
        "(https://open.dosm.gov.my/data-catalogue/hh_income_state)"
    )

st.caption(
    "JIE42403 Scientific Visualisation | "
    "OpenDOSM State-Level Statistics"
)
