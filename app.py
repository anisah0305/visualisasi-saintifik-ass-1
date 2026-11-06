import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.embed import file_html
from bokeh.resources import CDN

# ==========================================
# 1. DASHBOARD SETTINGS
# ==========================================

st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide"
)

GREEN = "#2D9B77"
DARK = "#176B52"
ORANGE = "#D88942"

sns.set_theme(style="whitegrid")

st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 1.4rem;
}
h1, h2, h3 {
    color: #176B52;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

@st.cache_data
def load_data():

    birth = pd.read_csv("birth_state.csv")
    income = pd.read_csv("hh_income_state.csv")
    school = pd.read_csv("completion_school_state.csv")

    for df in [birth, income, school]:
        df["state"] = df["state"].astype(str).str.strip()
        df["year"] = pd.to_datetime(
            df["date"], errors="coerce"
        ).dt.year

    for col in ["rate", "abs"]:
        birth[col] = pd.to_numeric(
            birth[col], errors="coerce"
        )

    for col in ["income_median", "income_mean"]:
        income[col] = pd.to_numeric(
            income[col], errors="coerce"
        )

    school["completion"] = pd.to_numeric(
        school["completion"], errors="coerce"
    )

    # Clean and remove duplicate observations
    birth = birth.dropna(
        subset=["year", "state"]
    ).drop_duplicates(["state", "year"])

    income = income.dropna(
        subset=["year", "state"]
    ).drop_duplicates(["state", "year"])

    # Exclude nationwide total
    school = school[
        school["state"] != "Malaysia"
    ].copy()

    school = school.dropna(
        subset=["year", "state", "stage", "sex"]
    ).drop_duplicates(
        ["state", "year", "stage", "sex"]
    )

    # Merge by state and year
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
    st.error(f"Could not read the CSV files: {error}")
    st.stop()


# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.markdown("""
<div style="
background:#176B52;
color:white;
padding:25px;
border-radius:16px;
">
    <div style="font-size:13px;color:#DDECE4">
        JIE42403 • SCIENTIFIC VISUALISATION
    </div>

    <div style="font-size:34px;font-weight:700">
        Malaysia Data Insights
    </div>

    <div>
        Live Births • Household Income • School Completion
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

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


# ==========================================
# 4. SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

years = sorted(
    set(bi.year.dropna()) & set(si.year.dropna())
)

year = st.sidebar.selectbox(
    "Income survey year",
    years,
    index=len(years) - 1
)

stages = {
    "secondary_upper": "Upper secondary",
    "secondary_lower": "Lower secondary",
    "primary": "Primary"
}

stage = st.sidebar.selectbox(
    "School stage",
    list(stages),
    format_func=lambda x: stages[x]
)

sexes = {
    "both": "Both genders",
    "female": "Female",
    "male": "Male"
}

sex = st.sidebar.selectbox(
    "Gender",
    list(sexes),
    format_func=lambda x: sexes[x]
)

incomes = {
    "income_median": "Median household income",
    "income_mean": "Mean household income"
}

income_field = st.sidebar.selectbox(
    "Income measure",
    list(incomes),
    format_func=lambda x: incomes[x]
)

st.sidebar.caption(
    "Income is available for selected survey years. "
    "Objective 3 covers annual data through 2025."
)


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def corr(df, x, y):

    d = df[[x, y]].dropna()

    if (
        len(d) >= 3
        and d[x].nunique() > 1
        and d[y].nunique() > 1
    ):
        return d[x].corr(d[y])

    return float("nan")


def meaning(r):

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
        direction = ""

    return strength + " " + direction


def show_plot(fig):

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


def summary_cards(values):

    cols = st.columns(len(values))

    for col, (name, value) in zip(cols, values):

        with col:
            st.metric(
                name,
                value,
                border=True
            )


def yearly_corr(df, x, y):

    result = []

    for yr, group in df.groupby("year"):

        result.append({
            "Year": int(yr),
            "Correlation": corr(group, x, y),
            "States": group.state.nunique()
        })

    return pd.DataFrame(result).sort_values("Year")


# ==========================================
# OBJECTIVE 1
# LIVE BIRTHS AND HOUSEHOLD INCOME
# ==========================================

def page1():

    st.header(
        "Objective 1: Annual Live Births and Household Income"
    )

    st.write(
        "**Objective:** To identify the relationship "
        "between annual live births and household income."
    )

    a = bi[
        bi.year == year
    ][[
        "state", "year", "rate", "abs", income_field
    ]].dropna().copy()

    r1 = corr(a, income_field, "rate")

    # SUMMARY BOXES
    summary_cards([
        ("Year", str(year)),
        ("States compared", str(a.state.nunique())),
        ("Pearson correlation", f"{r1:+.3f}")
    ])

    # CREATE INCOME GROUPS
    a["Income group"] = pd.qcut(
        a[income_field].rank(method="first"),
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

    # VISUALIZATION 1
    st.subheader(
        "Chart 1 — Average Birth Rate by Income Group"
    )

    fig, ax = plt.subplots(figsize=(9, 4.2))

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
            padding=3
        )

    ax.set(
        xlabel="Relative Income Group",
        ylabel="Births per 1,000 People",
        title=f"Birth Rate and Income Groups — {year}"
    )

    ax.set_ylim(0, grouped.rate.max() * 1.22)

    show_plot(fig)

    # TWO SUPPORTING VISUALIZATIONS
    left, right = st.columns(2)

    with left:

        st.subheader(
            "Chart 2 — Income and Birth Rate"
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.regplot(
            data=a,
            x=income_field,
            y="rate",
            ci=None,
            scatter_kws={
                "color": GREEN,
                "s": 70
            },
            line_kws={
                "color": ORANGE
            },
            ax=ax
        )

        ax.set(
            xlabel="Monthly Household Income (RM)",
            ylabel="Births per 1,000 People"
        )

        ax.xaxis.set_major_formatter(
            StrMethodFormatter("{x:,.0f}")
        )

        show_plot(fig)

    with right:

        st.subheader(
            "Chart 3 — Birth Rate by State"
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.barplot(
            data=a.sort_values("rate"),
            x="rate",
            y="state",
            color=GREEN,
            ax=ax
        )

        ax.set(
            xlabel="Births per 1,000 People",
            ylabel=""
        )

        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlim(left=0)

        show_plot(fig)

    # INTERPRETATION
    st.subheader("Interpretation and Discussion")

    st.write(
        f"In {year}, the relationship between household "
        f"income and birth rate is **{meaning(r1)}** "
        f"(Pearson r = **{r1:+.3f}**). "
        "The bar chart compares average birth rates across "
        "relative income groups, while the scatter plot "
        "shows the relationship across individual states. "
        "These associations do not prove causation."
    )

    # DATA PREVIEW
    with st.expander("Data Preview"):

        st.dataframe(
            a[
                ["state", "year", income_field, "rate", "abs"]
            ],
            hide_index=True
        )


# ==========================================
# OBJECTIVE 2
# SCHOOL COMPLETION AND HOUSEHOLD INCOME
# ==========================================

def page2():

    st.header(
        "Objective 2: School Completion Rates and Household Income"
    )

    st.write(
        "**Objective:** To examine the relationship "
        "between school completion rates and household income."
    )

    b = si[
        (si.year == year)
        & (si.stage == stage)
        & (si.sex == sex)
    ][[
        "state", "year", "completion", income_field
    ]].dropna().copy()

    r2 = corr(b, income_field, "completion")

    summary_cards([
        ("Year", str(year)),
        (
            "Average completion",
            f"{b.completion.mean():.1f}%"
        ),
        ("Pearson correlation", f"{r2:+.3f}")
    ])

    # VISUALIZATION 1
    st.subheader(
        "Chart 1 — Income versus School Completion"
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))

    sns.regplot(
        data=b,
        x=income_field,
        y="completion",
        ci=None,
        scatter_kws={
            "color": GREEN,
            "s": 80
        },
        line_kws={
            "color": ORANGE
        },
        ax=ax
    )

    ax.set(
        xlabel="Monthly Household Income (RM)",
        ylabel="School Completion Rate (%)",
        title=f"{stages[stage]} — {year}"
    )

    ax.xaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )

    show_plot(fig)

    left, right = st.columns(2)

    # VISUALIZATION 2
    with left:

        st.subheader(
            "Chart 2 — Completion by State"
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.barplot(
            data=b.sort_values("completion"),
            x="completion",
            y="state",
            color=GREEN,
            ax=ax
        )

        ax.set(
            xlabel="School Completion (%)",
            ylabel=""
        )

        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlim(left=0)

        show_plot(fig)

    # VISUALIZATION 3
    with right:

        st.subheader(
            "Chart 3 — Correlation by Survey Year"
        )

        trend = yearly_corr(
            si[
                (si.stage == stage)
                & (si.sex == sex)
            ],
            income_field,
            "completion"
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            trend.Year,
            trend.Correlation,
            color=GREEN,
            marker="o",
            linewidth=2.5
        )

        ax.axhline(
            0,
            color="gray",
            linestyle="--"
        )

        ax.set(
            xlabel="Income Survey Year",
            ylabel="Pearson Correlation (r)",
            ylim=(-1, 1)
        )

        ax.set_xticks(trend.Year.tolist())

        show_plot(fig)

    # INTERPRETATION
    st.subheader("Interpretation and Discussion")

    st.write(
        f"In {year}, **{stages[stage].lower()} school "
        f"completion** and household income show a "
        f"**{meaning(r2)}** relationship "
        f"(Pearson r = **{r2:+.3f}**). "
        "The scatter plot shows the overall association, "
        "while the state comparison highlights differences "
        "in completion rates. Other demographic, economic "
        "and educational factors may also be relevant."
    )

    with st.expander("Data Preview"):

        st.dataframe(
            b,
            hide_index=True
        )


# ==========================================
# OBJECTIVE 3
# LIVE BIRTHS AND SCHOOL COMPLETION
# ==========================================

def page3():

    st.header(
        "Objective 3: Annual Live Births and School Completion Rates"
    )

    st.write(
        "**Objective:** To analyse the relationship "
        "between annual live births and school completion rates."
    )

    c = bs[
        (bs.stage == stage)
        & (bs.sex == sex)
    ].dropna(
        subset=["rate", "completion"]
    ).copy()

    trend = yearly_corr(
        c,
        "rate",
        "completion"
    )

    selected = c[c.year == year]

    r3 = corr(
        selected,
        "rate",
        "completion"
    )

    summary_cards([
        ("Years analysed", str(trend.Year.nunique())),
        (
            "States compared",
            str(selected.state.nunique())
        ),
        (
            f"Correlation in {year}",
            f"{r3:+.3f}"
        )
    ])

    # VISUALIZATION 1 - BOKEH
    st.subheader(
        "Chart 1 — Interactive Yearly Correlation"
    )

    src = ColumnDataSource(trend)

    p = figure(
        title="Birth Rate and School Completion Correlation, 2016–2025",
        x_axis_label="Year",
        y_axis_label="Pearson Correlation (r)",
        width=980,
        height=400,
        x_range=(2015.5, 2025.5),
        y_range=(-1, 1),
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    p.line(
        x="Year",
        y="Correlation",
        source=src,
        color=DARK,
        line_width=3
    )

    dots = p.scatter(
        x="Year",
        y="Correlation",
        source=src,
        size=10,
        color=GREEN
    )

    p.add_tools(
        HoverTool(
            renderers=[dots],
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

    p.add_layout(
        Span(
            location=0,
            dimension="width",
            line_color="gray",
            line_dash="dashed"
        )
    )

    p.xaxis.ticker = list(range(2016, 2026))
    p.toolbar.logo = None

    st.iframe(
        file_html(p, CDN, "Yearly Correlation"),
        height=470,
        alt="Interactive Bokeh correlation line plot"
    )

    left, right = st.columns(2)

    # VISUALIZATION 2
    with left:

        st.subheader(
            f"Chart 2 — State Relationship ({year})"
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.regplot(
            data=selected,
            x="rate",
            y="completion",
            ci=None,
            scatter_kws={
                "color": GREEN,
                "s": 70
            },
            line_kws={
                "color": ORANGE
            },
            ax=ax
        )

        ax.set(
            xlabel="Births per 1,000 People",
            ylabel="School Completion (%)"
        )

        show_plot(fig)

    # VISUALIZATION 3
    with right:

        st.subheader(
            "Chart 3 — Correlation by School Stage"
        )

        rows = []

        for (yr, school_stage), part in bs[
            bs.sex == sex
        ].groupby(["year", "stage"]):

            rows.append({
                "year": int(yr),
                "stage": school_stage,
                "r": corr(
                    part,
                    "rate",
                    "completion"
                )
            })

        heat = pd.DataFrame(rows).pivot(
            index="stage",
            columns="year",
            values="r"
        )

        heat = heat.reindex([
            "primary",
            "secondary_lower",
            "secondary_upper"
        ])

        heat.index = [
            "Primary",
            "Lower secondary",
            "Upper secondary"
        ]

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            heat,
            center=0,
            vmin=-1,
            vmax=1,
            cmap="BrBG",
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={
                "label": "Pearson Correlation"
            },
            ax=ax
        )

        ax.set(
            xlabel="Year",
            ylabel=""
        )

        ax.tick_params(
            axis="x",
            labelrotation=45
        )

        show_plot(fig)

    # INTERPRETATION
    st.subheader("Interpretation and Discussion")

    st.write(
        f"The association between birth rate and "
        f"**{stages[stage].lower()} completion** in {year} "
        f"is **{meaning(r3)}** "
        f"(Pearson r = **{r3:+.3f}**). "
        "The line chart shows how this relationship "
        "changes over time, while the heatmap compares "
        "the association across educational stages. "
        "Births and school completion generally concern "
        "different cohorts, so these results should not "
        "be interpreted as causal effects."
    )

    with st.expander("Data Preview"):

        st.dataframe(
            selected[
                [
                    "state",
                    "year",
                    "rate",
                    "abs",
                    "completion"
                ]
            ],
            hide_index=True
        )


# ==========================================
# 6. PAGE NAVIGATION
# ==========================================

pages = st.navigation([
    st.Page(
        page1,
        title="01 • Births & Income",
        icon="📊",
        default=True
    ),
    st.Page(
        page2,
        title="02 • Education & Income",
        icon="🎓"
    ),
    st.Page(
        page3,
        title="03 • Births & Education",
        icon="📈"
    )
], position="sidebar")

pages.run()


# ==========================================
# 7. DATASET AND METHODOLOGY
# ==========================================

st.divider()

with st.expander(
    "Dataset Selection, Cleaning and Limitations"
):

    st.write(
        "**Source:** OpenDOSM state-level live births, "
        "household income and school completion datasets."
    )

    st.write(
        "**Data preparation:** State names and dates are "
        "standardized, duplicate observations are removed, "
        "and datasets are merged using matching states "
        "and years. Missing numerical values are not "
        "replaced with zero."
    )

    st.write(
        "**Limitations:** Household income data are "
        "available for selected years only. Income is "
        "reported in nominal RM, birth rates are measured "
        "per 1,000 population, and school completion "
        "rates can exceed 100%. Correlations represent "
        "state-level associations, not causal effects."
    )

    st.markdown(
        "[Births](https://open.dosm.gov.my/data-catalogue/births_annual_state)"
        " | "
        "[School Completion](https://open.dosm.gov.my/data-catalogue/completion_school_state)"
        " | "
        "[Household Income](https://open.dosm.gov.my/data-catalogue/hh_income_state)"
    )

st.caption(
    "JIE42403 Scientific Visualisation | "
    "Department of Statistics Malaysia (OpenDOSM)"
)
