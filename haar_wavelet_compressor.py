
import math
import numpy as np
import cv2

class HaarWaveletCompressor:
    def preprocess_image(self, img):
        max_dim_size = max(img.shape)
        N = 2 ** int(math.ceil(math.log(max_dim_size, 2)))
        return cv2.resize(img, (N, N))

    def get_transformation_matrix(self, k):
        np_trans_matrix = np.eye(2 ** k)
        for i in range(k):
            np_trans_matrix = np_trans_matrix @ self.get_part_of_transformation_matrix(i, k)
        return np_trans_matrix

    def get_part_of_transformation_matrix(self, part_idx: int, k: int) -> np.ndarray:
        transform = np.zeros((2 ** k, 2 ** k))
        for j in range(2 ** (k - part_idx - 1)):
            transform[2 * j, j] = 0.5
            transform[2 * j + 1, j] = 0.5
        offset = 2 ** (k - part_idx - 1)
        for j in range(2 ** (k - part_idx - 1)):
            transform[2 * j, offset + j] = 0.5
            transform[2 * j + 1, offset + j] = -0.5
        for j in range(2 ** (k - part_idx), 2 ** k):
            transform[j, j] = 1
        return transform
    
    def encode(self, img, t_matrix):
        encoded = t_matrix.T @ img @ t_matrix
        return encoded 
    
    def decode(self, img, t_matrix):
        t_matrix = np.linalg.inv(t_matrix)
        decoded = t_matrix.T @ img @ t_matrix
        return decoded

from matplotlib import pyplot as plt

def haar_wavelet_test():
    compressor = HaarWaveletCompressor()
    original_img = cv2.imread("images/test_cat.png")
    b_channel, g_channel, r_channel = cv2.split(original_img)
    b_image = compressor.preprocess_image(img=b_channel)
    g_image = compressor.preprocess_image(img=g_channel)
    r_image = compressor.preprocess_image(img=r_channel)
    b_trans_matrix = compressor.get_transformation_matrix(k=int(math.log2(len(b_image))))
    g_trans_matrix = compressor.get_transformation_matrix(k=int(math.log2(len(g_image))))
    r_trans_matrix = compressor.get_transformation_matrix(k=int(math.log2(len(r_image))))
    b_encoded = np.round(compressor.encode(img=b_image, t_matrix=b_trans_matrix))
    g_encoded = np.round(compressor.encode(img=g_image, t_matrix=g_trans_matrix))
    r_encoded = np.round(compressor.encode(img=r_image, t_matrix=r_trans_matrix))
    b_decoded = compressor.decode(img=b_encoded, t_matrix=b_trans_matrix)
    g_decoded = compressor.decode(img=g_encoded, t_matrix=g_trans_matrix)
    r_decoded = compressor.decode(img=r_encoded, t_matrix=r_trans_matrix)
    decoded_img = cv2.merge((b_decoded, g_decoded, r_decoded))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(original_img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Original") 
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(decoded_img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Decoded")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    haar_wavelet_test()