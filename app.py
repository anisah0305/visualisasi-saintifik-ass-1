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
# 1. PAGE SETTINGS AND DASHBOARD DESIGN
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
    max-width: 1150px;
    padding-top: 1.3rem;
}
h1, h2, h3 {
    color: #174737;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD, CLEAN AND MERGE DATASETS
# ==========================================

@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent

    birth = pd.read_csv(
        folder / "birth_state.csv"
    )

    income = pd.read_csv(
        folder / "hh_income_state.csv"
    )

    school = pd.read_csv(
        folder / "completion_school_state.csv"
    )

    # Standardize state names and dates
    for df in (birth, income, school):

        df["state"] = (
            df["state"].astype(str).str.strip()
        )

        df["year"] = pd.to_datetime(
            df["date"], errors="coerce"
        ).dt.year

        df.dropna(
            subset=["state", "year"],
            inplace=True
        )

        df["year"] = df["year"].astype(int)

        # Exclude nationwide totals
        df.drop(
            df[df["state"] == "Malaysia"].index,
            inplace=True
        )

    # Convert numerical columns
    for field in ("rate", "abs"):
        birth[field] = pd.to_numeric(
            birth[field], errors="coerce"
        )

    for field in ("income_median", "income_mean"):
        income[field] = pd.to_numeric(
            income[field], errors="coerce"
        )

    school["completion"] = pd.to_numeric(
        school["completion"], errors="coerce"
    )

    school["stage"] = (
        school["stage"].astype(str).str.strip()
    )

    school["sex"] = (
        school["sex"].astype(str).str.strip()
    )

    # Select study years and remove duplicates
    birth = birth[
        birth["year"].between(2016, 2025)
    ].drop_duplicates(["state", "year"])

    income = income[
        income["year"].between(2016, 2025)
    ].drop_duplicates(["state", "year"])

    school = school[
        school["year"].between(2016, 2025)
    ].drop_duplicates(
        ["state", "year", "stage", "sex"]
    )

    # Merge births and income
    bi = birth[
        ["state", "year", "rate", "abs"]
    ].merge(
        income[
            [
                "state", "year",
                "income_median", "income_mean"
            ]
        ],
        on=["state", "year"],
        how="inner",
        validate="one_to_one"
    )

    # Merge school completion and income
    si = school[
        [
            "state", "year", "stage",
            "sex", "completion"
        ]
    ].merge(
        income[
            [
                "state", "year",
                "income_median", "income_mean"
            ]
        ],
        on=["state", "year"],
        how="inner",
        validate="many_to_one"
    )

    # Merge births and school completion
    bs = birth[
        ["state", "year", "rate"]
    ].merge(
        school[
            [
                "state", "year", "stage",
                "sex", "completion"
            ]
        ],
        on=["state", "year"],
        how="inner",
        validate="one_to_many"
    )

    return bi, si, bs


try:
    bi, si, bs = load_data()

except Exception as exc:
    st.error(
        f"Unable to load datasets: {exc}"
    )
    st.stop()


# ==========================================
# 3. INTERACTIVE SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Analysis Filters")

valid_years = sorted(
    set(bi["year"]) & set(si["year"])
)

if not valid_years:
    st.error(
        "No overlapping income survey years were found."
    )
    st.stop()

year = st.sidebar.selectbox(
    "Income survey year",
    valid_years,
    index=len(valid_years) - 1
)

stages = {
    "secondary_upper": "Upper secondary",
    "secondary_lower": "Lower secondary",
    "primary": "Primary"
}

sexes = {
    "both": "Both genders",
    "female": "Female",
    "male": "Male"
}

incomes = {
    "income_median": "Median income",
    "income_mean": "Mean income"
}

stage = st.sidebar.selectbox(
    "School stage",
    list(stages),
    format_func=stages.get
)

sex = st.sidebar.selectbox(
    "Gender",
    list(sexes),
    format_func=sexes.get
)

income_col = st.sidebar.selectbox(
    "Monthly household income",
    list(incomes),
    format_func=incomes.get
)

st.sidebar.caption(
    "The year filter applies to Objectives 1 and 2. "
    "Objective 3 shows every year from 2016 to 2025."
)


# ==========================================
# 4. DASHBOARD HEADER AND OBJECTIVES
# ==========================================

st.markdown("""
<div style="
    background:#174737;
    color:white;
    padding:26px 28px;
    border-radius:16px;
    margin-bottom:16px;
">
    <div style="
        font-size:12px;
        letter-spacing:1.3px;
        color:#DDECE4;
        font-weight:700;
    ">
        JIE42403 · SCIENTIFIC VISUALISATION
    </div>

    <div style="
        font-size:33px;
        font-weight:750;
        margin-top:6px;
    ">
        Malaysia Data Insights
    </div>

    <div style="
        font-size:15px;
        color:#E4F2EC;
    ">
        Live Births · Household Income ·
        School Completion
    </div>
</div>
""", unsafe_allow_html=True)


with st.expander(
    "Research Objectives",
    expanded=True
):

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


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def pearson(data, x, y):

    valid = data[[x, y]].dropna()

    if (
        len(valid) < 3
        or valid[x].nunique() < 2
        or valid[y].nunique() < 2
    ):
        return float("nan")

    return float(valid[x].corr(valid[y]))


def relationship(r):

    if pd.isna(r):
        return "undefined"

    magnitude = abs(r)

    if magnitude < 0.10:
        strength = "negligible"
    elif magnitude < 0.30:
        strength = "weak"
    elif magnitude < 0.70:
        strength = "moderate"
    else:
        strength = "strong"

    if r > 0:
        direction = "positive"
    elif r < 0:
        direction = "negative"
    else:
        direction = "zero"

    return f"{strength} {direction}"


def show_cards(metrics):

    columns = st.columns(len(metrics))

    for column, (label, value) in zip(
        columns, metrics
    ):

        with column:
            st.metric(
                label,
                value,
                border=True
            )


def show_preview(data, columns):

    with st.expander(
        "Data Preview (Matched Observations)"
    ):

        st.dataframe(
            data[columns].sort_values("state"),
            hide_index=True,
            width="stretch"
        )

        st.caption(
            "These data have been cleaned and matched "
            "by state and year."
        )


# ==========================================
# 6. CREATE THREE OBJECTIVE TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "01 · Births & Income",
    "02 · Education & Income",
    "03 · Births & Education"
])


# ==========================================
# OBJECTIVE 1 - SEABORN BAR CHART
# ==========================================

with tab1:

    st.header(
        "Objective 1 — Annual Live Births "
        "and Household Income"
    )

    st.write(
        "**Objective:** To identify the relationship "
        "between annual live births and household income."
    )

    data1 = bi[
        bi["year"] == year
    ].dropna(
        subset=[income_col, "rate"]
    ).copy()

    if len(data1) < 4:

        st.warning(
            "Not enough matching observations for this year."
        )

    else:

        r1 = pearson(
            data1, income_col, "rate"
        )

        # IMPROVED SUMMARY BOXES
        show_cards([
            (
                "Average Birth Rate",
                f"{data1['rate'].mean():.2f} per 1,000"
            ),
            (
                "Pearson Correlation (r)",
                f"{r1:+.3f}"
            ),
            (
                "States Compared",
                f"{data1['state'].nunique()}"
            )
        ])

        st.caption(
            f"{year} · Relationship: {relationship(r1)} "
            "· Income categories are relative quartiles."
        )

        # Prepare four income groups
        data1["Income group"] = pd.qcut(
            data1[income_col].rank(
                method="first"
            ),
            q=4,
            labels=[
                "Lowest",
                "Lower-middle",
                "Upper-middle",
                "Highest"
            ]
        )

        grouped = data1.groupby(
            "Income group",
            observed=True
        )["rate"].mean().reset_index()

        # ONE VISUALIZATION
        st.subheader(
            "Average Birth Rate by "
            "Household Income Group"
        )

        fig1, ax1 = plt.subplots(
            figsize=(9, 4.5)
        )

        sns.barplot(
            data=grouped,
            x="Income group",
            y="rate",
            color=GREEN,
            ax=ax1
        )

        # Add values above bars
        for bars in ax1.containers:

            ax1.bar_label(
                bars,
                fmt="%.2f",
                padding=4,
                fontsize=10
            )

        ax1.set_xlabel(
            "Relative Household Income Group"
        )

        ax1.set_ylabel(
            "Live Births per 1,000 People"
        )

        # Start vertical axis at zero
        ax1.set_ylim(
            0,
            grouped["rate"].max() * 1.20
        )

        sns.despine(ax=ax1)
        fig1.tight_layout()

        st.pyplot(
            fig1,
            width="stretch"
        )

        plt.close(fig1)

        # Identify highest and lowest groups
        highest = grouped.loc[
            grouped["rate"].idxmax()
        ]

        lowest = grouped.loc[
            grouped["rate"].idxmin()
        ]

        # INTERPRETATION
        st.subheader(
            "Interpretation and Discussion"
        )

        st.write(
            f"In {year}, the **{highest['Income group']}** "
            "income group recorded the highest average "
            f"birth rate of **{highest['rate']:.2f} "
            "births per 1,000 people**, while the "
            f"**{lowest['Income group']}** group "
            "recorded the lowest average birth rate "
            f"of **{lowest['rate']:.2f}**. "
            "The Pearson correlation coefficient was "
            f"**{r1:+.3f}**, indicating a "
            f"**{relationship(r1)}** linear relationship. "
            "The results show an association but do "
            "not establish causation."
        )

        show_preview(
            data1,
            [
                "state", "year",
                income_col, "rate", "abs"
            ]
        )


# ==========================================
# OBJECTIVE 2 - SEABORN SCATTER PLOT
# ==========================================

with tab2:

    st.header(
        "Objective 2 — School Completion "
        "Rates and Household Income"
    )

    st.write(
        "**Objective:** To examine the relationship "
        "between school completion rates and household income."
    )

    data2 = si[
        (si["year"] == year)
        & (si["stage"] == stage)
        & (si["sex"] == sex)
    ].copy()

    data2 = data2.dropna(
        subset=[income_col, "completion"]
    )

    if (
        len(data2) < 3
        or data2[income_col].nunique() < 2
    ):

        st.warning(
            "Not enough valid observations "
            "for the selected filters."
        )

    else:

        r2 = pearson(
            data2,
            income_col,
            "completion"
        )

        # IMPROVED SUMMARY BOXES
        show_cards([
            (
                "Average Completion Rate",
                f"{data2['completion'].mean():.2f}%"
            ),
            (
                "Pearson Correlation (r)",
                f"{r2:+.3f}"
            ),
            (
                "States Compared",
                f"{data2['state'].nunique()}"
            )
        ])

        st.caption(
            f"{year} · {stages[stage]} · {sexes[sex]} "
            f"· Relationship: {relationship(r2)}"
        )

        # ONE VISUALIZATION
        st.subheader(
            "Household Income versus "
            "School Completion Rate"
        )

        fig2, ax2 = plt.subplots(
            figsize=(9, 4.7)
        )

        sns.regplot(
            data=data2,
            x=income_col,
            y="completion",
            ci=None,
            scatter_kws={
                "color": GREEN,
                "s": 85,
                "alpha": 0.85,
                "edgecolor": "white"
            },
            line_kws={
                "color": ORANGE,
                "linewidth": 2.5
            },
            ax=ax2
        )

        ax2.set_xlabel(
            "Monthly Household Income (RM)"
        )

        ax2.set_ylabel(
            "School Completion Rate (%)"
        )

        ax2.xaxis.set_major_formatter(
            StrMethodFormatter("{x:,.0f}")
        )

        sns.despine(ax=ax2)
        fig2.tight_layout()

        st.pyplot(
            fig2,
            width="stretch"
        )

        plt.close(fig2)

        # INTERPRETATION
        st.subheader(
            "Interpretation and Discussion"
        )

        st.write(
            f"For **{stages[stage].lower()}**, "
            f"**{sexes[sex].lower()}**, in {year}, "
            "the Pearson correlation coefficient "
            f"was **{r2:+.3f}**, indicating a "
            f"**{relationship(r2)}** linear relationship "
            "between household income and school "
            "completion rates. Each point represents "
            "one Malaysian state or federal territory. "
            "The orange regression line illustrates "
            "the overall direction of the relationship. "
            "Other socioeconomic and educational factors "
            "may also influence school completion outcomes."
        )

        show_preview(
            data2,
            [
                "state", "year",
                income_col, "completion"
            ]
        )


# ==========================================
# OBJECTIVE 3 - BOKEH INTERACTIVE LINE CHART
# ==========================================

with tab3:

    st.header(
        "Objective 3 — Annual Live Births "
        "and School Completion Rates"
    )

    st.write(
        "**Objective:** To analyse the relationship "
        "between annual live births and school completion rates."
    )

    data3 = bs[
        (bs["stage"] == stage)
        & (bs["sex"] == sex)
    ].dropna(
        subset=["rate", "completion"]
    ).copy()

    # Calculate yearly Pearson correlations
    rows = []

    for yr, group in data3.groupby("year"):

        r = pearson(
            group,
            "rate",
            "completion"
        )

        if pd.notna(r):

            rows.append({
                "Year": int(yr),
                "Correlation": r,
                "States": group["state"].nunique()
            })

    trend = pd.DataFrame(rows)

    if trend.empty:

        st.warning(
            "No valid yearly correlations "
            "for this selection."
        )

    else:

        trend = trend.sort_values(
            "Year"
        ).reset_index(drop=True)

        latest = trend.iloc[-1]

        # Find strongest positive correlation
        positive = trend[
            trend["Correlation"] > 0
        ]

        if positive.empty:

            best_label = "Strongest Positive r"
            best_value = "None"

        else:

            best = positive.loc[
                positive["Correlation"].idxmax()
            ]

            best_label = (
                "Strongest Positive r "
                f"({int(best['Year'])})"
            )

            best_value = (
                f"{best['Correlation']:+.3f}"
            )

        # IMPROVED SUMMARY BOXES
        show_cards([
            (
                "Years Analysed",
                f"{len(trend)}"
            ),
            (
                best_label,
                best_value
            ),
            (
                f"Latest Pearson r ({int(latest['Year'])})",
                f"{latest['Correlation']:+.3f}"
            )
        ])

        st.caption(
            f"{stages[stage]} · {sexes[sex]} "
            "· Latest relationship: "
            f"{relationship(latest['Correlation'])}"
        )

        # ONE VISUALIZATION
        st.subheader(
            "Yearly Birth Rate and "
            "School Completion Relationship"
        )

        source = ColumnDataSource(trend)

        plot = figure(
            x_axis_label="Year",
            y_axis_label=(
                "Pearson Correlation Coefficient (r)"
            ),
            height=415,
            sizing_mode="stretch_width",
            x_range=(2015.5, 2025.5),
            y_range=(-1, 1),
            tools=(
                "pan,wheel_zoom,"
                "box_zoom,reset,save"
            )
        )

        # Draw line
        plot.line(
            "Year",
            "Correlation",
            source=source,
            line_width=3,
            color=DARK
        )

        # Draw interactive markers
        points = plot.scatter(
            "Year",
            "Correlation",
            source=source,
            size=11,
            color=GREEN
        )

        # Add hover tooltips
        plot.add_tools(
            HoverTool(
                renderers=[points],
                tooltips=[
                    ("Year", "@Year"),
                    (
                        "Pearson r",
                        "@Correlation{0.000}"
                    ),
                    ("States", "@States")
                ]
            )
        )

        # Reference line at zero
        plot.add_layout(
            Span(
                location=0,
                dimension="width",
                line_color="#888888",
                line_dash="dashed"
            )
        )

        plot.xaxis.ticker = list(
            range(2016, 2026)
        )

        plot.ygrid.grid_line_alpha = 0.20
        plot.toolbar.logo = None

        # Display interactive Bokeh chart
        st.iframe(
            file_html(
                plot,
                CDN,
                "Births and School Completion"
            ),
            height=475,
            alt=(
                "Interactive Bokeh line chart "
                "showing yearly Pearson correlations"
            )
        )

        # INTERPRETATION
        st.subheader(
            "Interpretation and Discussion"
        )

        st.write(
            "The relationship between birth rates "
            f"and **{stages[stage].lower()} school "
            "completion** changed from "
            f"{int(trend['Year'].min())} to "
            f"{int(trend['Year'].max())}. "
            f"In {int(latest['Year'])}, the Pearson "
            "correlation coefficient was "
            f"**{latest['Correlation']:+.3f}**, "
            "indicating a "
            f"**{relationship(latest['Correlation'])}** "
            "linear relationship. Positive values "
            "indicate positive associations, while "
            "negative values indicate negative "
            "associations. The findings represent "
            "state-level statistical relationships "
            "and do not demonstrate causation."
        )

        # DATA PREVIEW
        with st.expander(
            "Yearly Correlation Data"
        ):

            st.dataframe(
                trend.round(3),
                hide_index=True,
                width="stretch"
            )


# ==========================================
# 7. DATASET SOURCES AND METHODOLOGY
# ==========================================

st.divider()

with st.expander(
    "Dataset Sources, Preparation and Limitations"
):

    st.write(
        "**Sources:** OpenDOSM state-level annual "
        "live births, household income and school "
        "completion datasets."
    )

    st.write(
        "**Data preparation:** Dates were converted "
        "to years, state names were standardized, "
        "numerical fields were converted, duplicate "
        "observations were removed, national totals "
        "were excluded, and the datasets were merged "
        "using matching states and years."
    )

    st.write(
        "**Missing data:** Missing numerical values "
        "were not replaced with zero. Only valid "
        "observations were included in each analysis."
    )

    st.write(
        "**Limitations:** Household income data are "
        "available for selected survey years only. "
        "Income is reported in nominal RM. "
        "School completion rates may exceed 100% "
        "under the official indicator definition. "
        "The results describe statistical associations "
        "rather than causal effects."
    )

    st.markdown(
        "[Live Births]"
        "(https://open.dosm.gov.my/data-catalogue/births_annual_state)"
        " · "
        "[School Completion]"
        "(https://open.dosm.gov.my/data-catalogue/completion_school_state)"
        " · "
        "[Household Income]"
        "(https://open.dosm.gov.my/data-catalogue/hh_income_state)"
    )

st.caption(
    "JIE42403 Scientific Visualisation · "
    "OpenDOSM · One Visualization per Objective"
)
