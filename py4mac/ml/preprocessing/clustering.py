import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

def tsne_clustering(data, len, n_components=2, random_state=42):
    tsne = manifold.TSNE(n_components=n_components, random_state=random_state)
    transformed_data = tsne.fit_transform(data[:len, :])
    tsne_df = pd.DataFrame(
        np.column_stack((transformed_data, targets[: 3000])),
        columns=["x", "y", "targets"]
    )
    tsne_df.loc[:, 'targets'] = tsne_df.targets.astype(int)
    return tsne_df

def plt_tsne(tsne_df, size=8):
    grid = sns.FacetGrid(tsne_df, hue="targets", size=size)
    grid.map(plt.scatter, "x", "y").add_legend()
