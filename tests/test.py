# %% import libraries and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from libs.word2vec.model import Word2Vec

# Loading training data
input_text = np.concatenate([np.arange(30) for _ in range(100)]).astype(str) #in this case, since the array is 1d, axis=0 is the same as None, which means the array is flattened before the operation is performed.

# %% Training
model = Word2Vec(window=5, vector_size=32, epochs=5)
model.fit(input_text)
emb = model.transform()

# %% Visualize the embedding
# %% 
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(7, 5))
xy = PCA(n_components=2).fit_transform(emb)

sns.scatterplot(x=xy[:, 0], y=xy[:, 1], ax=ax, s=10, alpha=0.5)

output_folder = "files"
output_file = output_folder + "word2vec_embedding.png"
fig.savefig(output_file, bbox_inches="tight", dpi=300)
