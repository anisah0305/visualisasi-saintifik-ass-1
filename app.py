import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.embed import file_html
from bokeh.resources import CDN

# 1. APPLICATION SETTINGS
st.set_page_config(
    page_title="Malaysia Scientific Visualisation",
    page_icon="📊",
    layout="wide"
)

GREEN = "#2D9B77"
DARK_GREEN = "#176B52"
ORANGE = "#D88942"

sns.set_theme(style="whitegrid")

st.title("Malaysia: Births, Education & Household Income")
st.caption("JIE42403 | Scientific Visualisation")
st.write(
    "An interactive analysis of Malaysian state-level "
    "birth rates, school completion and household income."
)


# 2. LOAD THE CLEANED DATASETS
@st.cache_data
def load_data():
    obj1 = pd.read_csv("objective1_clean.csv")
    obj2 = pd.read_csv("objective2_clean.csv")
    obj3 = pd.read_csv("objective3_clean.csv")

    for df in [obj1, obj2, obj3]:
        df["year"] = df["year"].astype(int)

    return obj1, obj2, obj3


obj1, obj2, obj3 = load_data()


# 3. YEAR FILTER
available_years = sorted(
    set(obj1["year"]) & set(obj2["year"])
)

st.sidebar.header("Dashboard Filters")

selected_year = st.sidebar.selectbox(
    "Select Income Survey Year",
    available_years,
    index=len(available_years) - 1
)

st.sidebar.info(
    "Household income is available for selected "
    "survey years. Objective 3 covers 2016–2025."
)

# Filter data for the selected year
data1 = obj1[
    obj1["year"] == selected_year
].dropna(subset=["income_median", "rate"]).copy()

data2 = obj2[
    obj2["year"] == selected_year
].dropna(
    subset=["income_median", "completion"]
).copy()


# 4. INTERPRETATION HELPER
def explain_correlation(r):
    if pd.isna(r):
        return "undefined"

    if abs(r) < 0.10:
        strength = "negligible"
    elif abs(r) < 0.30:
        strength = "weak"
    elif abs(r) < 0.70:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


# 5. CREATE THREE TABS
tab1, tab2, tab3 = st.tabs([
    "Objective 1: Births & Income",
    "Objective 2: Education & Income",
    "Objective 3: Births & Education"
])


# OBJECTIVE 1 - SEABORN BAR CHART
with tab1:
    st.header("Annual Live Births and Household Income")

    st.write(
        "**Objective:** To identify the relationship "
        "between annual live births and household income."
    )

    correlation1 = data1["rate"].corr(
        data1["income_median"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Year", selected_year)
    c2.metric("States Compared", data1["state"].nunique())
    c3.metric("Pearson Correlation", f"{correlation1:+.3f}")

    # Divide states into four income groups
    data1["Income Group"] = pd.qcut(
        data1["income_median"],
        q=4,
        labels=[
            "Low Income",
            "Lower-Middle",
            "Upper-Middle",
            "High Income"
        ]
    )

    summary1 = data1.groupby(
        "Income Group", observed=True
    )["rate"].mean().reset_index()

    # Seaborn bar chart
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=summary1,
        x="Income Group",
        y="rate",
        color=GREEN,
        ax=ax1
    )

    for container in ax1.containers:
        ax1.bar_label(
            container, fmt="%.2f", padding=4
        )

    ax1.set_title(
        f"Average Birth Rate by Income Group ({selected_year})"
    )
    ax1.set_xlabel("Household Income Group")
    ax1.set_ylabel("Birth Rate per 1,000 People")
    ax1.set_ylim(0, summary1["rate"].max() * 1.2)

    plt.tight_layout()
    st.pyplot(fig1, width="stretch")
    plt.close(fig1)

    st.subheader("Interpretation")
    st.write(
        f"The state-level correlation for {selected_year} "
        f"is {correlation1:+.3f}, indicating a "
        f"{explain_correlation(correlation1)} relationship. "
        "The bars compare average birth rates across "
        "four relative household-income groups."
    )


# OBJECTIVE 2 - SEABORN SCATTER PLOT
with tab2:
    st.header("School Completion and Household Income")

    st.write(
        "**Objective:** To examine the relationship "
        "between school completion rates and household income."
    )

    correlation2 = data2["income_median"].corr(
        data2["completion"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Year", selected_year)
    c2.metric(
        "Average Completion",
        f"{data2['completion'].mean():.2f}%"
    )
    c3.metric("Pearson Correlation", f"{correlation2:+.3f}")

    # Seaborn regression scatter plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sns.regplot(
        data=data2,
        x="income_median",
        y="completion",
        ci=None,
        scatter_kws={
            "color": GREEN,
            "s": 100,
            "alpha": 0.85
        },
        line_kws={
            "color": ORANGE,
            "linewidth": 2.5
        },
        ax=ax2
    )

    ax2.set_title(
        f"Household Income vs School Completion ({selected_year})"
    )
    ax2.set_xlabel("Median Monthly Household Income (RM)")
    ax2.set_ylabel("Upper-Secondary Completion Rate (%)")

    ax2.xaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )

    plt.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close(fig2)

    st.subheader("Interpretation")
    st.write(
        f"The Pearson correlation is {correlation2:+.3f}, "
        f"indicating a {explain_correlation(correlation2)} "
        "relationship between household income and "
        "upper-secondary school completion. "
        "Each point represents a state or federal territory."
    )


# OBJECTIVE 3 - BOKEH INTERACTIVE LINE CHART
with tab3:
    st.header("Annual Live Births and School Completion")

    st.write(
        "**Objective:** To analyse the relationship "
        "between annual live births and school completion rates."
    )

    # Calculate correlations across states each year
    yearly_results = []

    for year, group in obj3.groupby("year"):
        group = group.dropna(
            subset=["rate", "completion"]
        )

        if (
            len(group) >= 3
            and group["rate"].nunique() > 1
            and group["completion"].nunique() > 1
        ):
            yearly_results.append({
                "Year": int(year),
                "Correlation": group["rate"].corr(
                    group["completion"]
                ),
                "States": group["state"].nunique()
            })

    yearly = pd.DataFrame(yearly_results)
    yearly = yearly.sort_values("Year")

    selected_r = yearly.loc[
        yearly["Year"] == selected_year,
        "Correlation"
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Years Analysed", len(yearly))
    c2.metric("States Compared", obj3["state"].nunique())
    c3.metric(
        f"{selected_year} Correlation",
        f"{selected_r:+.3f}"
    )

    source = ColumnDataSource(yearly)

    # Create Bokeh figure
    p = figure(
        title="Yearly Birth Rate and School Completion Correlation",
        x_axis_label="Year",
        y_axis_label="Pearson Correlation (r)",
        width=850,
        height=430,
        x_range=(2015.5, 2025.5),
        y_range=(-1, 1),
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    p.line(
        x="Year",
        y="Correlation",
        source=source,
        line_width=3,
        color=DARK_GREEN
    )

    points = p.scatter(
        x="Year",
        y="Correlation",
        source=source,
        size=10,
        color=GREEN
    )

    # Add hover tooltips
    p.add_tools(HoverTool(
        renderers=[points],
        tooltips=[
            ("Year", "@Year"),
            ("Correlation", "@Correlation{0.000}"),
            ("States", "@States")
        ]
    ))

    # Add zero correlation reference
    p.add_layout(Span(
        location=0,
        dimension="width",
        line_color="gray",
        line_dash="dashed"
    ))

    p.xaxis.ticker = list(range(2016, 2026))
    p.ygrid.grid_line_alpha = 0.25
    p.toolbar.logo = None

    # Display interactive Bokeh HTML
    chart_html = file_html(
        p, CDN, "Birth Rate and School Completion"
    )

    st.iframe(
        chart_html,
        height=500,
        alt="Interactive yearly correlation line chart"
    )

    st.subheader("Interpretation")
    st.write(
        "The line shows how the correlation between "
        "birth rates and upper-secondary completion "
        "varies across states from 2016 to 2025. "
        f"In {selected_year}, the correlation is "
        f"{selected_r:+.3f}, indicating a "
        f"{explain_correlation(selected_r)} relationship."
    )


# 6. DATA SOURCE INFORMATION
st.divider()

st.caption(
    "Data Source: OpenDOSM | Malaysia State-Level Statistics"
)

st.info(
    "These visualisations show statistical associations, "
    "not evidence that one variable causes another."
)
