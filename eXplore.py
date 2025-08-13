import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# --------------------
# Load data from GitHub
# --------------------
@st.cache_data
def load_data():
    features_url = "https://raw.githubusercontent.com/Sibabrata29/poleXplore_new/main/polyeXplore%20model%20feature%20%26%20index.xlsx"
    library_url = "https://raw.githubusercontent.com/Sibabrata29/poleXplore_new/main/polyeXplore%20polymer%20library%20data.xlsx"

    features = pd.read_excel(features_url, sheet_name="polyFeature", index_col=0)
    index_df = pd.read_excel(features_url, sheet_name="polyIndex", index_col=0)
    library = pd.read_excel(library_url, sheet_name="reference", index_col=0)
    return features, index_df, library

features, index_df, library = load_data()

# --------------------
# App Title
# --------------------
st.title("Polymer eXplore")

feature_names = list(features.columns)
valid_polymers = list(features.index)

# --------------------
# User input form
# --------------------
with st.form("polymer_form"):
    polymer_name_input = st.text_input("Enter the polymer name:").strip()

    # Collect feature scores
    user_features = []
    for name in feature_names:
        val = st.number_input(f"Enter your feature score for {name}:", value=0.0, format="%.2f")
        user_features.append(val)

    submit_button = st.form_submit_button("Submit")

# --------------------
# After submit
# --------------------
if submit_button:
    # Validation for alphabets only
    if not polymer_name_input.isalpha():
        st.error("❌ Polymer name should contain only alphabets.")
    elif polymer_name_input not in valid_polymers:
        st.error("❌ Polymer not found in database. Please enter a valid polymer name.")
    else:
        # Convert user input to array
        user_sf_array = np.array(user_features).reshape(1, -1)

        # Compute user PPW
        user_ppw = user_sf_array.dot(index_df.values)

        # Compute PPW for database
        ppw_df = pd.DataFrame(features.values.dot(index_df.values), index=features.index, columns=index_df.columns)

        # Find 2 nearest neighbors (based on PPW)
        nbrs_ppw = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nbrs_ppw.fit(ppw_df.values)
        distances_ppw, indices_ppw = nbrs_ppw.kneighbors(user_ppw)

        nearest_polymers = [features.index[idx] for idx in indices_ppw[0]]
        distances_list = distances_ppw[0]

        # --------------------
        # Feature Comparison Table
        # --------------------
        comp_df = pd.DataFrame({polymer_name_input: user_features}, index=feature_names)
        for p in nearest_polymers:
            comp_df[p] = features.loc[p].values

        st.subheader("Structural Feature Comparison")
        st.dataframe(comp_df)

        # --------------------
        # Distance Bar Chart
        # --------------------
        colors = ["green" if i == 0 else "gold" for i in range(len(nearest_polymers))]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(nearest_polymers, distances_list, color=colors)
        ax.set_xlabel("Euclidean Distance (PPW Space)")
        ax.set_title(f"Similarity to '{polymer_name_input}'")
        ax.invert_yaxis()

        for bar, dist in zip(bars, distances_list):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{dist:.2f}", va='center')

        st.pyplot(fig)

        # --------------------
        # Reference Properties for Matches
        # --------------------
        st.subheader("Reference Properties of Matches")
        st.dataframe(library.loc[nearest_polymers])
