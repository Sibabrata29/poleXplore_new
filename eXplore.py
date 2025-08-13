import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from tabulate import tabulate
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Load Data from GitHub
# ------------------------------
@st.cache_data
def load_data():
    features_url = "https://raw.githubusercontent.com/Sibabrata29/poleXplore_new/main/polyeXplore%20model%20feature%20%26%20index.xlsx"
    library_url = "https://raw.githubusercontent.com/Sibabrata29/poleXplore_new/main/polyeXplore%20polymer%20library%20data.xlsx"

    features = pd.read_excel(features_url, sheet_name="polyFeature", index_col=0)
    index_df = pd.read_excel(features_url, sheet_name="polyIndex", index_col=0)
    library = pd.read_excel(library_url, sheet_name="reference", index_col=0)

    return features, index_df, library

features, index_df, library = load_data()

# ------------------------------
# Step 2: User Polymer Name Input
# ------------------------------
st.title("Polymer Structural Feature Explorer")

valid_polymers = list(features.index)

while True:
    polymer_name_input = st.text_input("Enter the polymer name:").strip()
    if polymer_name_input:
        if not polymer_name_input.isalpha():
            st.error("Polymer name should contain only alphabets.")
        elif polymer_name_input not in valid_polymers:
            st.error("Polymer not found in dataset. Please enter a valid polymer name.")
        else:
            break
    st.stop()

# ------------------------------
# Step 3: User Feature Scores Input
# ------------------------------
feature_names = [
    "Chain Flexibility", "Rigidity hetero Linkage", "Aromaticity", "Side Group Rigidity",
    "Chain Packing", "Functional Groups", "H-Bonding", "pi-pi Stacking", "Fused Rings", "Toughening factor"
]

user_features = []
st.subheader(f"Enter your feature scores for: {polymer_name_input}")
for name in feature_names:
    val = st.number_input(f"{name}:", value=0.0, step=0.5, format="%.2f")
    user_features.append(val)

user_sf_array = np.array(user_features).reshape(1, -1)

# ------------------------------
# Step 4: Calculate User PPW (Hidden)
# ------------------------------
user_ppw = user_sf_array.dot(index_df.values)
user_ppw_array = np.array(user_ppw).reshape(1, -1)

# ------------------------------
# Step 5: Find 2 Nearest Neighbours
# ------------------------------
nbrs_ppw = NearestNeighbors(n_neighbors=2, metric='euclidean')
nbrs_ppw.fit(features.values.dot(index_df.values))
distances_ppw, indices_ppw = nbrs_ppw.kneighbors(user_ppw_array)

nearest_polymers = [features.index[idx] for idx in indices_ppw[0]]
distances_list = distances_ppw[0]

# ------------------------------
# Step 6: Feature Comparison Table
# ------------------------------
comparison_df = pd.DataFrame(
    {polymer_name_input: user_features,
     nearest_polymers[0]: features.loc[nearest_polymers[0]],
     nearest_polymers[1]: features.loc[nearest_polymers[1]]},
    index=feature_names
)

st.subheader("Structural Feature Comparison")
st.dataframe(comparison_df)

# ------------------------------
# Step 7: Euclidean Distance Bar Chart
# ------------------------------
st.subheader("Similarity to User Polymer (Lower = More Similar)")

colors = ["green", "gold"]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.barh(nearest_polymers, distances_list, color=colors)
ax.set_xlabel("Euclidean Distance (Property Space)")
ax.invert_yaxis()

for bar, dist in zip(bars, distances_list):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{dist:.2f}", va='center')

st.pyplot(fig)

# ------------------------------
# Step 8: Reference Properties for Matches
# ------------------------------
st.subheader("Reference Properties for Nearest Polymers")
properties_df = library.loc[nearest_polymers].T
st.markdown(tabulate(properties_df, headers='keys', tablefmt='pipe'))
