import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="Movie Rating Prediction Dashboard",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("movies_cleaned.csv")
    return df

df = load_data()

# ---------------------------------------------------------
# Language code cleanup
# ---------------------------------------------------------

language_map = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "hi": "Hindi",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "cn": "Cantonese/Chinese",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",
    "he": "Hebrew",
    "th": "Thai",
    "id": "Indonesian",
    "fa": "Persian",
    "cs": "Czech",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "vi": "Vietnamese",
    "uk": "Ukrainian"
}

if "original_language" in df.columns:
    df["language_name"] = df["original_language"].map(language_map).fillna(df["original_language"])
else:
    df["language_name"] = "Unknown"

# ---------------------------------------------------------
# Main title
# ---------------------------------------------------------

st.title("🎬 Predicting Movie Audience Ratings")
st.markdown(
    """
    This Streamlit application demonstrates a predictive analytics solution using the Kaggle Movies Dataset.
    The goal is to help movie studios, streaming platforms, and investors estimate audience rating risk using
    production characteristics and early performance indicators.
    """
)

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Movie Filters")

df_filtered = df.copy()

if "release_year" in df.columns:
    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())

    year_range = st.sidebar.slider(
        "Release Year Range",
        min_year,
        max_year,
        (min_year, max_year)
    )

    df_filtered = df_filtered[
        (df_filtered["release_year"] >= year_range[0]) &
        (df_filtered["release_year"] <= year_range[1])
    ]

if "is_english" in df.columns:
    language_group = st.sidebar.selectbox(
        "Language Group",
        ["All Movies", "English-Language Movies", "Non-English-Language Movies"]
    )

    if language_group == "English-Language Movies":
        df_filtered = df_filtered[df_filtered["is_english"] == 1]
    elif language_group == "Non-English-Language Movies":
        df_filtered = df_filtered[df_filtered["is_english"] == 0]

if "budget" in df.columns:
    max_budget = int(df["budget"].max())

    budget_range = st.sidebar.slider(
        "Budget Range",
        0,
        max_budget,
        (0, max_budget)
    )

    df_filtered = df_filtered[
        (df_filtered["budget"] >= budget_range[0]) &
        (df_filtered["budget"] <= budget_range[1])
    ]

if "runtime" in df.columns:
    min_runtime = int(df["runtime"].min())
    max_runtime = int(df["runtime"].max())

    runtime_range = st.sidebar.slider(
        "Runtime Range",
        min_runtime,
        max_runtime,
        (min_runtime, max_runtime)
    )

    df_filtered = df_filtered[
        (df_filtered["runtime"] >= runtime_range[0]) &
        (df_filtered["runtime"] <= runtime_range[1])
    ]

if "vote_average" in df.columns:
    rating_range = st.sidebar.slider(
        "Audience Rating Range",
        0.0,
        10.0,
        (0.0, 10.0)
    )

    df_filtered = df_filtered[
        (df_filtered["vote_average"] >= rating_range[0]) &
        (df_filtered["vote_average"] <= rating_range[1])
    ]

if "vote_count" in df.columns:
    max_votes = int(df["vote_count"].max())

    vote_count_range = st.sidebar.slider(
        "Vote Count Range",
        0,
        max_votes,
        (0, max_votes)
    )

    df_filtered = df_filtered[
        (df_filtered["vote_count"] >= vote_count_range[0]) &
        (df_filtered["vote_count"] <= vote_count_range[1])
    ]

st.sidebar.divider()
st.sidebar.write("Full dataset rows:", len(df))
st.sidebar.write("Filtered rows:", len(df_filtered))

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "Data Explorer",
    "Insights Dashboard",
    "Rating Prediction Tool",
    "Business Recommendations"
])

# ---------------------------------------------------------
# Tab 1: Executive Overview
# ---------------------------------------------------------

with tab1:
    st.header("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Movies Analyzed", f"{len(df_filtered):,}")

    with col2:
        if "vote_average" in df_filtered.columns and len(df_filtered) > 0:
            st.metric("Average Rating", round(df_filtered["vote_average"].mean(), 2))
        else:
            st.metric("Average Rating", "N/A")

    with col3:
        if "budget" in df_filtered.columns and len(df_filtered) > 0:
            st.metric("Average Budget", f"${df_filtered['budget'].mean():,.0f}")
        else:
            st.metric("Average Budget", "N/A")

    with col4:
        if "revenue" in df_filtered.columns and len(df_filtered) > 0:
            st.metric("Average Revenue", f"${df_filtered['revenue'].mean():,.0f}")
        else:
            st.metric("Average Revenue", "N/A")

    st.subheader("Business Problem")
    st.write(
        """
        Movie studios and streaming platforms make high-cost decisions before knowing how audiences will respond.
        This project uses predictive analytics to estimate audience ratings earlier in the decision-making process,
        helping decision makers identify potential rating risk and support better planning.
        """
    )

    st.subheader("Research Question")
    st.info(
        "Can a movie’s average audience rating be predicted using production characteristics and early performance indicators?"
    )

    st.subheader("Analytical Approach")
    st.write(
        """
        This project uses a regression-based predictive model. The target variable is audience rating,
        represented by `vote_average`. Predictor variables include budget, revenue, popularity, vote count,
        runtime, release timing, language, genre count, and other engineered features.
        """
    )

    st.subheader("Model Summary")
    st.write(
        """
        The baseline model using only budget had very weak explanatory power, with adjusted R² around 0.014.
        After feature engineering and adding stronger predictors such as popularity, vote count, revenue, runtime,
        and production-related features, the model improved substantially, reaching adjusted R² around 0.438
        with RMSE around 0.60.
        """
    )

# ---------------------------------------------------------
# Tab 2: Data Explorer
# ---------------------------------------------------------

with tab2:
    st.header("Data Explorer")

    st.write(
        """
        This section allows users to explore the cleaned movie dataset after applying the sidebar filters.
        The full raw Kaggle dataset was reduced to a deployment-ready version containing only the fields needed
        for modeling, dashboarding, and interpretation.
        """
    )

    if len(df_filtered) == 0:
        st.warning("No movies match the current filters. Adjust the sidebar filters to view data.")
    else:
        st.dataframe(df_filtered.head(100), use_container_width=True)

        st.subheader("Dataset Summary")
        st.dataframe(df_filtered.describe(include="all"), use_container_width=True)

    st.subheader("Columns Used in the Project")
    st.write(
        """
        The app uses production characteristics, early engagement indicators, and engineered features such as
        log-transformed budget, revenue, popularity, and vote count. These features help convert raw movie data
        into a more useful predictive structure.
        """
    )

# ---------------------------------------------------------
# Tab 3: Insights Dashboard
# ---------------------------------------------------------

with tab3:
    st.header("Insights Dashboard")

    if len(df_filtered) == 0:
        st.warning("No data available for the current filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if "release_year" in df_filtered.columns and "vote_average" in df_filtered.columns:
                yearly_rating = df_filtered.groupby("release_year", as_index=False)["vote_average"].mean()

                fig = px.line(
                    yearly_rating,
                    x="release_year",
                    y="vote_average",
                    title="Average Audience Rating by Release Year"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "runtime" in df_filtered.columns and "vote_average" in df_filtered.columns:
                fig = px.scatter(
                    df_filtered,
                    x="runtime",
                    y="vote_average",
                    title="Runtime vs. Audience Rating",
                    opacity=0.5
                )
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            if "budget" in df_filtered.columns and "vote_average" in df_filtered.columns:
                fig = px.scatter(
                    df_filtered,
                    x="budget",
                    y="vote_average",
                    title="Budget vs. Audience Rating",
                    opacity=0.5
                )
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            if "popularity" in df_filtered.columns and "vote_average" in df_filtered.columns:
                fig = px.scatter(
                    df_filtered,
                    x="popularity",
                    y="vote_average",
                    title="Popularity vs. Audience Rating",
                    opacity=0.5
                )
                st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)

        with col5:
            if "vote_count" in df_filtered.columns and "vote_average" in df_filtered.columns:
                fig = px.scatter(
                    df_filtered,
                    x="vote_count",
                    y="vote_average",
                    title="Vote Count vs. Audience Rating",
                    opacity=0.5
                )
                st.plotly_chart(fig, use_container_width=True)

        with col6:
            if "language_name" in df_filtered.columns and "vote_average" in df_filtered.columns:
                language_rating = (
                    df_filtered.groupby("language_name", as_index=False)
                    .agg(
                        average_rating=("vote_average", "mean"),
                        movie_count=("vote_average", "count")
                    )
                )

                language_rating = language_rating[language_rating["movie_count"] >= 10]
                language_rating = language_rating.sort_values("average_rating", ascending=False).head(10)

                fig = px.bar(
                    language_rating,
                    x="language_name",
                    y="average_rating",
                    title="Top Languages by Average Rating",
                    hover_data=["movie_count"]
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Insight")
        st.success(
            """
            Budget alone is not enough to explain audience rating. Stronger predictive value comes from combining
            production variables with early engagement indicators such as popularity and vote count.
            """
        )

# ---------------------------------------------------------
# Tab 4: Rating Prediction Tool
# ---------------------------------------------------------

with tab4:
    st.header("Rating Prediction Tool")

    st.write(
        """
        Enter a movie scenario below. The app will estimate the expected audience rating using a regression model
        trained on the cleaned movie dataset.
        """
    )

    model_features = [
        "log_budget",
        "log_revenue",
        "runtime",
        "release_year",
        "release_month",
        "genre_count",
        "production_company_count",
        "language_count",
        "is_english",
        "has_tagline",
        "log_popularity",
        "log_vote_count"
    ]

    available_features = [col for col in model_features if col in df.columns]

    if "vote_average" in df.columns and len(available_features) >= 3:
        model_data = df[available_features + ["vote_average"]].dropna()

        X = model_data[available_features]
        y = model_data["vote_average"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=474
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Model R²", round(r2, 3))

        with col2:
            st.metric("RMSE", round(rmse, 3))

        st.subheader("Enter Movie Inputs")

        col_a, col_b = st.columns(2)

        with col_a:
            budget = st.number_input(
                "Budget",
                min_value=0,
                value=10000000,
                step=1000000
            )

            revenue = st.number_input(
                "Revenue",
                min_value=0,
                value=50000000,
                step=1000000
            )

            runtime = st.number_input(
                "Runtime",
                min_value=1,
                value=110,
                step=1
            )

            release_year = st.number_input(
                "Release Year",
                min_value=1900,
                max_value=2030,
                value=2020,
                step=1
            )

            release_month = st.slider(
                "Release Month",
                1,
                12,
                6
            )

            popularity = st.number_input(
                "Popularity",
                min_value=0.0,
                value=20.0,
                step=1.0
            )

        with col_b:
            vote_count = st.number_input(
                "Vote Count",
                min_value=0,
                value=1000,
                step=100
            )

            genre_count = st.slider(
                "Number of Genres",
                1,
                8,
                2
            )

            production_company_count = st.slider(
                "Number of Production Companies",
                0,
                20,
                2
            )

            language_count = st.slider(
                "Number of Spoken Languages",
                1,
                10,
                1
            )

            is_english_input = st.selectbox(
                "Is the Movie Originally in English?",
                ["Yes", "No"]
            )

            has_tagline_input = st.selectbox(
                "Does the Movie Have a Tagline?",
                ["Yes", "No"]
            )

        input_values = {
            "log_budget": np.log1p(budget),
            "log_revenue": np.log1p(revenue),
            "runtime": runtime,
            "release_year": release_year,
            "release_month": release_month,
            "genre_count": genre_count,
            "production_company_count": production_company_count,
            "language_count": language_count,
            "is_english": 1 if is_english_input == "Yes" else 0,
            "has_tagline": 1 if has_tagline_input == "Yes" else 0,
            "log_popularity": np.log1p(popularity),
            "log_vote_count": np.log1p(vote_count)
        }

        input_df = pd.DataFrame([input_values])
        input_df = input_df[available_features]

        if st.button("Predict Audience Rating"):
            prediction = model.predict(input_df)[0]

            prediction = max(0, min(10, prediction))

            st.subheader("Prediction Result")
            st.metric("Predicted Audience Rating", round(prediction, 2))

            if prediction >= 6.8:
                st.success("Strong expected rating. This movie profile appears relatively favorable.")
            elif prediction >= 5.8:
                st.warning("Moderate expected rating. This movie profile may perform acceptably but has some uncertainty.")
            else:
                st.error("Lower expected rating. This movie profile may carry audience rating risk.")

            st.write(
                """
                A non-technical stakeholder can use this prediction as an early signal, not as a guaranteed outcome.
                The model supports better planning by identifying whether a movie profile appears stronger, moderate,
                or potentially risky based on historical patterns.
                """
            )

        st.subheader("How to Interpret This Tool")
        st.info(
            """
            This prediction should be interpreted as a decision-support estimate. It does not guarantee audience reaction,
            but it helps identify whether a movie profile looks stronger or weaker compared to historical patterns.
            """
        )

    else:
        st.warning(
            """
            The prediction tool needs the required engineered features and `vote_average` column.
            Make sure your cleaned dataset includes the model variables used in your final project.
            """
        )

# ---------------------------------------------------------
# Tab 5: Business Recommendations
# ---------------------------------------------------------

with tab5:
    st.header("Business Recommendations")

    st.subheader("Decision-Maker Value")
    st.write(
        """
        This solution helps decision makers evaluate movie projects earlier by translating historical movie data
        into a practical prediction and risk-assessment tool. Instead of relying only on intuition, executives can
        compare movie profiles, test assumptions, and identify potential rating risk before making marketing,
        distribution, or investment decisions.
        """
    )

    st.subheader("Recommended Use Cases")
    st.markdown(
        """
        - Compare expected ratings across different movie profiles.
        - Identify movies that may need stronger marketing support.
        - Evaluate how early popularity and engagement indicators relate to audience response.
        - Support greenlighting, distribution, and campaign planning decisions.
        - Communicate model findings to non-technical stakeholders through a simple web dashboard.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - The model does not perfectly predict audience taste.
        - Vote count and popularity may reflect post-release engagement, so predictions are strongest when early signals are available.
        - External factors such as reviews, actors, franchise strength, and cultural trends may also influence ratings.
        - The model should support decision-making rather than replace executive judgment.
        """
    )

    st.subheader("Final Learning")
    st.info(
        """
        The main takeaway is that audience ratings are partially predictable, but no single variable explains them.
        The strongest solution comes from combining business understanding, data preparation, feature engineering,
        model evaluation, and deployment into one accessible decision-support tool.
        """
    )
