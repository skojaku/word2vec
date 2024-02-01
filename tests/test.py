# %%
import numpy as np
import pandas as pd

# Training data
input_text = np.concatenate([np.arange(30) for _ in range(100)]).astype(str)

# %% Training
model = Word2Vec(window=5, vector_size=32, epochs=10)
model.fit(input_text)
emb = model.transform()

# %% Visualize the embedding
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))


xy = PCA(n_components=2).fit_transform(emb)

sns.scatterplot(x=xy[:, 0], y=xy[:, 1], ax=ax, s=10, alpha=0.5)

output_file = "word2vec_embedding.png"
fig.savefig(output_file, bbox_inches="tight", dpi=300)
