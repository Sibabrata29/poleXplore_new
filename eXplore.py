import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# ====== Load Data ======
@st.cache_data
def load_data():
    features_url = "https://raw.githubusercontent.com/Sibabrata29/poleXplore_new/main/polyeXplore%20model%20feature%20%26%20index.xlsx"
    library_url = "https://raw.githubusercontent.com/Sibabrata29/poleXplore_new/main/polyeXplore%20polymer%20library%20data.xlsx"

    features = pd.read_excel(features_url, sheet_name="polyFeature", index_col=0)
    index_df = pd.read_excel(features_url, sheet_name="polyIndex", index_col=0)
    library = pd.read_excel(library_url, sheet_name="reference", index_col=0)

    return features, index_df, library

features, index_df, library = load_data()

# ====== User Input ======
st.title("Polymer Structural Feature Matching Tool")

valid_polymers = list(features.index)
polymer_name = st.selectbox("Select Polymer from Database:", valid_polymers)

st.write(f"### Polymer Investigated: **{polymer_name}**")

feature_names = features.columns.tolist()
user_features = []
for name in feature_names:
    val = st.number_input(f"{name}:", value=float(features.loc[polymer_name, name]), step=0.1)
    user_features.append(val)

user_sf_array = np.array(user_features).reshape(1, -1)

# ====== Nearest Neighbor (PPW space) ======
user_ppw = user_sf_array.dot(index_df.values)
ppw_df = pd.DataFrame(features.values.dot(index_df.values),
                      index=features.index,
                      columns=index_df.columns)

nbrs_ppw = NearestNeighbors(n_neighbors=2, metric='euclidean')
nbrs_ppw.fit(ppw_df.values)
distances, indices = nbrs_ppw.kneighbors(user_ppw)

nearest_polymers = [features.index[i] for i in indices[0]]

# ====== Distance Bar Chart ======
distances_list = [0.0] + list(distances[0])
bar_labels = [polymer_name] + nearest_polymers
colors = ["green", "gold", "red"]

fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
bars = ax_bar.barh(bar_labels, distances_list, color=colors)
ax_bar.set_xlabel("Euclidean Distance (Property Space)")
ax_bar.set_title(f"Similarity to '{polymer_name}'")
ax_bar.invert_yaxis()
for bar, dist in zip(bars, distances_list):
    ax_bar.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{dist:.2f}", va='center')
st.pyplot(fig_bar)

# ====== Feature Comparison ======
compare_df = pd.DataFrame({polymer_name: user_features}, index=feature_names)
for p in nearest_polymers:
    compare_df[p] = features.loc[p].values

st.subheader("Structural Feature Comparison")
st.dataframe(compare_df)

# ====== Grouped Bar Chart ======
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(compare_df.index))
width = 0.8 / len(compare_df.columns)
for i, col in enumerate(compare_df.columns):
    ax.bar(x + i*width, compare_df[col], width, label=col)
ax.set_ylabel("Feature Score")
ax.set_xticks(x + width*(len(compare_df.columns)-1)/2)
ax.set_xticklabels(compare_df.index, rotation=45, ha="right")
ax.legend()
st.pyplot(fig)

# ====== Display Library Properties ======
properties_df = library.loc[nearest_polymers].T
properties_df.columns.name = None
st.subheader("Reference Properties of Nearest Matches")
st.dataframe(properties_df)
