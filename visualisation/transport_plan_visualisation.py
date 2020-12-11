import matplotlib.pyplot as plt

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