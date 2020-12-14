import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.spatial.distance import cdist
import numpy as np

def imshow_transportation(G, cmap = 'RdBu'):
    '''
    G: ndarray (num_i, num_f)
        the transportation plan
    '''
    plt.figure(figsize = (9, 9))
    plt.imshow(G, cmap = cmap)
    plt.axis('off')

def plot_transportation_2d(x_i, x_f, G, line_color = 'k', src_color = 'blue', dst_color = 'red', with_labels = False):
    '''
    x_i: ndarray (num_i, 2)
        the source
    x_f: ndarray (num_f, 2)
        the sink
    G: ndarray (num_i, num_f)
        the transportation plan
    '''
    assert(x_i.shape[1] == 2 and x_f.shape[1] == 2)
    
    plt.figure(figsize = (9, 6))
    
    plt.scatter(x_i[:, 0], x_i[:, 1], color = src_color, marker = 'o', label = "src")
    plt.scatter(x_f[:, 0], x_f[:, 1], color = dst_color, marker = 'x', label = "dst")
    
    max_G = G.max()
    
    for i, src in enumerate(x_i):
        for j, dst in enumerate(x_f):
            dx = dst[0] - src[0]
            dy = dst[1] - src[1]
            plt.arrow(src[0], src[1], dx, dy, color = line_color, alpha = G[i, j] / max_G)
            
    if with_labels:
        plt.legend(fontsize = 18)
            
    plt.axis("off")
    
def plot_text_transportation(s1, s2, s1_vec, s2_vec, G):
    '''
    s1: array-like (n, )
        words from source sentence
    s2: array-like (m, )
        words from target sentence
    s1_vec: ndarray (n, n_features)
        vectorization of s1
    s2_vec: ndarray (m, n_features)
        vectorization of s2        
    G: ndarray (n, m)
        the transportation plan
    '''
    C = cdist(np.concatenate((s1_vec, s2_vec)),
              np.concatenate((s1_vec, s2_vec)))
    
    nmds = manifold.MDS(
        2,
        eps=1e-9,
        dissimilarity="precomputed",
        n_init=1)
    npos = nmds.fit_transform(C)
    
    n = s1_vec.shape[0]

    plt.figure(figsize=(6,6))
    plt.scatter(npos[:n,0],npos[:n,1],c='r',s=50, edgecolor = 'k')
    for i, txt in enumerate(s1):
        plt.annotate(txt, (npos[i,0],npos[i,1]),fontsize=20)
    plt.scatter(npos[n:,0],npos[n:,1],c='b',s=50, edgecolor = 'k')
    for i, txt in enumerate(s2):
        plt.annotate(txt, (npos[i+n,0],npos[i+n,1]),fontsize=20)


    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[i,j]>1e-5:
                plt.plot([npos[i,0],npos[j+n,0]],[npos[i,1],npos[j+n,1]],
                         'k', alpha=G[i,j] / np.max(G))

    plt.axis('off')
    plt.tight_layout()
