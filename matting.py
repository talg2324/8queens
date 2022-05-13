import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange

def main():

    scale = {'queen':0.1, 'board':0.3}

    for im_name in ['queen', 'board']:
        # Already done
        if not os.path.exists('./images/%s_matted.jpg' %im_name):
            if not os.path.exists('./images/%s_trimap.npy' %im_name):
                im = load_im('./images/%s.jpg' %im_name, scale[im_name])
                trimap = trimapify(im, im_name)
            else:
                im = load_im('./images/%s.jpg' %im_name, scale[im_name])
                trimap = np.load('./images/%s_trimap.npy' %im_name)
            refine_trimap(trimap, './images/%s_alphamap.npy' %im_name)
            finalize_matting(im, trimap, im_name)

def finalize_matting(im, trimap, im_name):
    outpath = './images/%s_matted.jpg' %im_name

    im[trimap==0, :] = 0
    cv2.imwrite(outpath, im)

def refine_trimap(trimap, path):
    binary_im = np.zeros(trimap.shape, dtype=np.uint8)
    binary_im[trimap>0] = 255
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_im, connectivity=8)
    
    largest_component = np.argmax(stats[1:, -1])+1
    trimap[output != largest_component] = np.minimum(trimap[output != largest_component], 0)
    
    # Refine the unknown points
    # Start at the farthest point and work inwards
    CoM = centroids[largest_component, :]
    y0, x0 = np.where(trimap==0)
    R = np.hypot(y0-CoM[1], x0-CoM[0])
    coord_sorted = np.stack((R, y0, x0), axis=-1).astype(np.int32)
    coord_sorted = coord_sorted[np.argsort(coord_sorted[:, 0])[::-1]]

    for r, py, px in coord_sorted:
        neighborhood_vote = int(np.sum(trimap[py-1:py+2, px-1:px+2])>0) # 8-neighbor vote
        trimap[py, px] = 2 * neighborhood_vote - 1

    trimap[trimap<0] = 0
    # Save the alpha map
    np.save(path, trimap)

    plt.figure()
    plt.imshow(trimap, cmap='gray')
    plt.axis('off')
    plt.savefig('./samples/' + path.split('/')[-1][:-4] + '.png', bbox_inches='tight')
    return trimap

def trimapify(im, im_name):
    foreground, background = initial_segmentation(im)

    # Probability map
    gmm = TrivialGMM(im, foreground, background)
    foreground_prob, background_prob = gmm(im)

    Df = calc_distance_map(foreground, foreground_prob)
    Db = calc_distance_map(background, background_prob)
    
    trimap = get_trimap(Df, Db, 5)
    
    
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(foreground, cmap='hot')
    plt.axis('off')
    plt.tick_params(bottom=False)
    plt.subplot(1,4,3)
    plt.imshow(Df, cmap='hot')
    plt.axis('off')
    plt.subplot(1,4,4)
    plt.imshow(trimap, cmap='jet')
    plt.axis('off')
    plt.savefig('./samples/%s_matting.png'%im_name, bbox_inches='tight')

    outpath = './images/%s_trimap.npy' %im_name
    np.save(outpath, trimap)
    return trimap

def initial_segmentation(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    foreground = np.zeros_like(im_gray)
    foreground[im_gray < 100] = 255
    background = np.zeros_like(foreground)
    background[-50:, ...] = 255
    return foreground, background

def calc_distance_map(mask, probabilities):

    # Init
    start_y, start_x = np.where(mask==255)
    distance_map = np.zeros_like(probabilities) + 1e6
    visited = np.zeros_like(mask)
    distance_map[start_y, start_x] = 0
    queue = [(start_y[i], start_x[i]) for i in range(start_y.shape[0])]

    djikstra(queue, visited, distance_map, probabilities, mask.shape)
    return np.log(distance_map + 1e-7)

def djikstra(queue, visited, distance_map, probabilities, shape):
    
    maxy = shape[0]-2
    maxx = shape[1]-2

    while queue:
        i, j = queue.pop()

        bottom = max(0, i-1)
        top = min(maxy, i+2)
        left = max(0, j-1)
        right = min(maxx, j+2)

        for ii in prange(bottom, top):
            for jj in prange(left, right):
                W = abs(probabilities[ii, jj] - probabilities[i, j])
                distance_map[ii, jj] = min(distance_map[ii, jj], distance_map[i, j] + W)
                    
                if not visited[ii,jj]:
                    queue.append((ii, jj))

        visited[i, j] = 1

def get_trimap(Df, Db, delta=0.5):
    """
    Create a distance based trimap
    Each pixel value is:
    1 - Foreground
    0 - Unsure
    -1 - Background
    """

    trimap = np.abs(Df-Db) * np.argmin([Db, Df], axis=0) # Looking for argmin
    trimap[trimap==0] = -1
    trimap[(trimap > 0) & (trimap < delta)] = 0
    trimap[trimap > delta] = 1

    return trimap

class TrivialGMM():
    def __init__(self, X, foreground_mask, background_mask):

        foreground = X[foreground_mask==255, :]
        background = X[background_mask==255, :]

        self.mu = np.vstack([np.mean(foreground, axis=0), np.mean(background, axis=0)])
        self.cov_inv = np.array([np.linalg.inv(np.cov(foreground, rowvar=False)), np.linalg.inv(np.cov(background, rowvar=False))])

    def __call__(self, Y):
        foreground = np.zeros((Y.shape[0], Y.shape[1]), dtype=np.float32)
        background = np.zeros((Y.shape[0], Y.shape[1]), dtype=np.float32)
        fast_prob_estimate(self.mu, self.cov_inv, Y.astype(np.float32), foreground, background, Y.shape)
        denom = foreground + background

        foreground[denom.nonzero()] /= denom[denom.nonzero()]
        background[denom.nonzero()] /= denom[denom.nonzero()]

        return cv2.GaussianBlur(foreground, (5,5), 1.0), cv2.GaussianBlur(background, (5,5), 1.0)

@njit(parallel=True)
def fast_prob_estimate(mu, cov_inv, inp, foreground, background, shape):
    for i in prange(shape[0]):
        for j in prange(shape[1]):

            pixel = inp[i,j, :]

            # Foreground
            d = mu[0, :] - pixel
            mahalanobis = d @ cov_inv[0, ...] @ d.T
            foreground[i,j] = mahalanobis

            # Background
            d = mu[1, :] - pixel
            mahalanobis = d @ cov_inv[1, ...] @ d.T
            background[i,j] = mahalanobis

def load_im(path, scale):
    im = cv2.imread(path)
    im = cv2.resize(im, None, fx=scale, fy=scale)
    return im


if __name__ == "__main__":
    main()