
import os
import argparse

import math
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt

from huffman_compressor import HuffmanCompressor
from yuv_chroma_compressor import ChromaSubsamplingCompressor
from haar_wavelet_compressor import HaarWaveletCompressor

class MyCompressor:
    def __init__(self, yuv420 = False, haar = False, huff = False):
        self.yuv420 = yuv420
        self.haar = haar
        self.huff = huff

    def encode(self, image_path):
        compressed_data = dict()
        image = cv2.imread(image_path)
        if self.yuv420: image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        c1, c2, c3 = cv2.split(image)
        if self.yuv420:
            c2 = self.yuv_channel_encode(channel=c2)
            c3 = self.yuv_channel_encode(channel=c3)
        if self.haar:
            c1 = self.haar_channel_encode(channel=c1)
            c2 = self.haar_channel_encode(channel=c2)
            c3 = self.haar_channel_encode(channel=c3)
            compressed_data['haar_shape'] = np.array(list(c1)).shape
        if self.huff:
            c1, f1 = self.huff_channel_encode(channel=c1)
            c2, f2 = self.huff_channel_encode(channel=c2)
            c3, f3 = self.huff_channel_encode(channel=c3)
            compressed_data['frequencies_1'] = f1
            compressed_data['frequencies_2'] = f2
            compressed_data['frequencies_3'] = f3
        compressed_data['channel_1'] = np.array(list(c1))
        compressed_data['channel_2'] = np.array(list(c2))
        compressed_data['channel_3'] = np.array(list(c3))
        compressed_data['original_shape'] = (image.shape[0], image.shape[1])
        np.savez_compressed('compressed_data.npz', **compressed_data)

    def decode(self, compressed_file_path):
        compressed_data = np.load(compressed_file_path, allow_pickle=True)
        c1 = compressed_data['channel_1']
        c2 = compressed_data['channel_2']
        c3 = compressed_data['channel_3']
        shape = compressed_data['haar_shape'] if self.haar else compressed_data['original_shape']
        if self.huff:
            f1 = compressed_data['frequencies_1'].item()
            f2 = compressed_data['frequencies_2'].item()
            f3 = compressed_data['frequencies_3'].item()
            c1 = self.huff_channel_decode(channel=c1, frequencies=f1, shape=shape)
            c2 = self.huff_channel_decode(channel=c2, frequencies=f2, shape=(-1, shape[1] // 2) if self.yuv420 else shape)
            c3 = self.huff_channel_decode(channel=c3, frequencies=f3, shape=(-1, shape[1] // 2) if self.yuv420 else shape)
        if self.haar:
            c1 = self.haar_channel_decode(channel=c1, shape=shape)
            c2 = self.haar_channel_decode(channel=c2, shape=(-1, shape[1] // 2) if self.yuv420 else shape)
            c3 = self.haar_channel_decode(channel=c3, shape=(-1, shape[1] // 2) if self.yuv420 else shape)
        if self.yuv420:
            c2 = self.yuv_channel_decode(channel=c2, shape=shape)
            c3 = self.yuv_channel_decode(channel=c3, shape=shape)
        decoded_data = cv2.merge((c1, c2, c3)).clip(0, 255).astype(np.uint8)
        if self.yuv420: decoded_data = cv2.cvtColor(decoded_data, cv2.COLOR_YUV2BGR)
        if self.haar: decoded_data = cv2.resize(decoded_data, compressed_data['original_shape'][::-1])
        cv2.imwrite("decoded_image.png", decoded_data)

    def haar_channel_encode(self, channel):
        haar = HaarWaveletCompressor()
        image = haar.preprocess_image(img=channel)
        trans_matrix = haar.get_transformation_matrix(k=int(math.log2(len(image))))
        encoded_data = haar.encode(img=image, t_matrix=trans_matrix)
        return np.round(encoded_data)
    
    def haar_channel_decode(self, channel, shape):
        haar = HaarWaveletCompressor()
        trans_matrix = haar.get_transformation_matrix(k=int(math.log2(shape[1])))
        decoded = haar.decode(img=channel, t_matrix=trans_matrix)
        return decoded
    
    def yuv_channel_encode(self, channel):
        yuv = ChromaSubsamplingCompressor()
        encoded_data = yuv.encode(channel=channel)
        return encoded_data
    
    def yuv_channel_decode(self, channel, shape):
        yuv = ChromaSubsamplingCompressor()
        decoded_data = yuv.decode(channel=channel, o_shape=shape)
        return decoded_data

    def huff_channel_encode(self, channel):
        frequencies = defaultdict(int)
        for row in channel:
            for pixel in row:
                frequencies[pixel] += 1
        huff = HuffmanCompressor()
        tree_root = huff.build_tree(frequencies=frequencies)
        codes = huff.build_codes(tree_root)
        encoded_data = huff.encode(channel.flatten(), codes)
        return encoded_data, frequencies

    def huff_channel_decode(self, channel, frequencies, shape):
        huff = HuffmanCompressor()
        tree_root = huff.build_tree(frequencies)
        decoded_data = np.array(list(huff.decode(channel, tree_root))).reshape(shape)
        return decoded_data

if __name__ == '__main__':

    # Handle the consol arguments

    parser = argparse.ArgumentParser(description="Compress and decompress an image")
    parser.add_argument('-i', '--img', action='store', type=str, default="images/test_cat.png", help="Path of the target image")
    parser.add_argument('--yuv420', action='store_true', help="Convert to YUV and use chroma subsampling (4:2:0)")
    parser.add_argument('--haar', action='store_true', help="Use Haar Wavelet Transformation with value rounding")
    parser.add_argument('--huff', action='store_true', help="Use Huffman coding and decoding (loosless)")
    args = parser.parse_args()

    # Compress and decompress the image

    compressor = MyCompressor(yuv420=args.yuv420, haar=args.haar, huff=args.huff)
    compressor.encode(image_path=args.img)
    compressor.decode(compressed_file_path='compressed_data.npz')

    # Calculate the MSE and the PSNR values

    original_img = cv2.imread(args.img)
    decoded_img = cv2.imread('decoded_image.png')

    mse = np.mean((original_img - decoded_img) ** 2)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse != 0 else float('inf')

    # Print the results

    original_img_bytes = os.path.getsize(args.img)
    encoded_img_bytes = os.path.getsize('compressed_data.npz')
    decoded_img_bytes = os.path.getsize('decoded_image.png')

    print("\n= Results ==================")
    print(f" - MSE:    {mse:10.2f}")
    print(f" - PSNR:   {psnr:10.2f}")
    print(f" - O Size: {original_img_bytes:10} bytes")
    print(f" - E Size: {encoded_img_bytes:10} bytes")
    print(f" - D Size: {decoded_img_bytes:10} bytes")
    print("============================\n")

    # Plot the original and the decoded image

    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original") 
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB))
    plt.title("Decoded")
    plt.axis('off')

    plt.show()
    