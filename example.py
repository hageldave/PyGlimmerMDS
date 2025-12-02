import matplotlib.pyplot as plt
import numpy as np
from pyglimmermds import Glimmer, execute_glimmer

def show_curr_state(glimmer_state, labels, i):
    lvl = glimmer_state["level"]
    itr = glimmer_state["iter"]
    emb = glimmer_state["embedding"]
    idx = glimmer_state["index_set"]
    cur = emb[idx]
    lab = labels[idx]
    fig,ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f"Glimmer MDS - level: {lvl} - iter: {itr:03} ")
    ax.scatter(emb[:, 0], emb[:, 1], c="black", s=0.02)
    colors = ["limegreen", "dodgerblue", "orange"]
    for l in range(3):
        lbl_idx = np.where(lab==l)[0]
        ax.scatter(cur[lbl_idx, 0], cur[lbl_idx, 1], c=colors[l], s=0.03*(lvl*2+1))
    fig.savefig(f"fig_out/{i:04}.png")
    plt.close(fig)


def animate_glimmer(dataset: dict, mds: Glimmer):
    iter = [0]
    def callback(obj):
        show_curr_state(obj, labels, iter[0])
        iter[0]+=1
    mds.callback = callback
    #execute_glimmer(dataset['data'], callback=callback, rng=rng)
    projection = mds.fit_transform(dataset['data'])


if __name__ == '__main__':
    from sklearn import preprocessing as prep
    from sklearn import datasets

    rng = np.random.default_rng(seed=0xC0FFEE)

    # get iris data
    dataset = datasets.load_iris()
    data = dataset.data
    labels = dataset.target
    # duplicate data with added noise
    for _ in range(8):
        data = np.vstack((data, data + (rng.random((data.shape[0], data.shape[1])) * 0.2 - .1)))
        labels = np.append(labels, labels)
    data = prep.StandardScaler().fit_transform(data)
    print(data.shape)

    dataset = dict(data=data, labels=labels)
    mds = Glimmer(rng=rng)
    animate_glimmer(dataset, mds)

    