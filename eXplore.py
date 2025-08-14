import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="polyeXplore prediction model", layout="wide")
st.title("Polymer structural features & property index influence matrix")


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

# ---------- USER INPUT ----------
poly_name = st.text_input("Enter polymer name to explore (case-sensitive):", "")

feature_names = [
    "Chain Flexibility", "Rigidity hetero Linkage", "Aromaticity",
    "Side Group Rigidity", "Chain Packing", "Functional Groups",
    "H-Bonding", "pi-pi Stacking", "Fused Rings", "Toughening factor"
]

user_features = []
if poly_name:
    if poly_name not in features.index:
        st.error("Polymer name not found in dataset.")
    else:
        st.subheader("Enter Feature Scores")
        for f in feature_names:
            val = st.number_input(f"{f}:", min_value=-5.0, max_value=10.0, step=0.5)
            user_features.append(val)

        if st.button("Run Assessment"):
            # User data arrays
            user_sf_array = np.array(user_features).reshape(1, -1)
            user_ppw_array = user_sf_array.dot(index_df.values)

            # Find 2 nearest neighbours in PPW space
            nbrs_ppw = NearestNeighbors(n_neighbors=2, metric='euclidean')
            nbrs_ppw.fit(ppw_df.values)
            distances_ppw, indices_ppw = nbrs_ppw.kneighbors(user_ppw_array)

            # ---------- Comparison Table ----------
            comparison_df = pd.DataFrame({f"Data: {poly_name}": features.loc[poly_name].values},
                                          index=feature_names)
            comparison_df[f"User Entry: {poly_name}"] = user_features

            for idx in indices_ppw[0]:
                nn_name = features.index[idx]
                comparison_df[nn_name] = features.loc[nn_name].values

            st.markdown("### Structural Feature Comparison")
            st.dataframe(comparison_df)

            # ---------- Horizontal Bar Chart ----------
            fig, ax = plt.subplots(figsize=(10, 6))
            y = np.arange(len(comparison_df.index))
            height = 0.8 / len(comparison_df.columns)
            for i, col in enumerate(comparison_df.columns):
                ax.barh(y + i*height, comparison_df[col], height, label=col)
            ax.set_yticks(y + height*(len(comparison_df.columns)-1)/2)
            ax.set_yticklabels(comparison_df.index)
            ax.set_xlabel("Feature Score")
            ax.set_title(f"Structural Comparison for '{poly_name}'")
            ax.legend()
            st.pyplot(fig)

            # ---------- Nearest neighbour distances ----------
            nearest_polymers = [features.index[idx] for idx in indices_ppw[0]]
            distances_display = [0] + list(distances_ppw[0])
            labels_display = [poly_name] + nearest_polymers
            colors = ["green"] + ["gold"]*len(nearest_polymers)

            fig2, ax2 = plt.subplots()
            bars = ax2.barh(labels_display, distances_display, color=colors)
            ax2.invert_yaxis()
            ax2.set_xlabel("Euclidean Distance (Property Space)")
            ax2.set_title("Nearest Neighbours")
            for bar, dist in zip(bars, distances_display):
                ax2.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{dist:.2f}", va='center')
            st.pyplot(fig2)

            # ---------- Reference properties ----------
            props_df = library.loc[nearest_polymers].T
            st.markdown("### Reference Properties for Nearest Polymers")
            st.table(props_df)
