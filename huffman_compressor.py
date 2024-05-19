
from bitarray import bitarray
from heapq import heappush, heappop, heapify

class HuffmanNode:
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

class HuffmanCompressor:
    def build_tree(self, frequencies):
        heap = [HuffmanNode(symbol, freq) for symbol, freq in frequencies.items()]
        heapify(heap)
        while len(heap) > 1:
            left = heappop(heap)
            right = heappop(heap)
            merged = HuffmanNode(frequency=left.frequency + right.frequency)
            merged.left = left
            merged.right = right
            heappush(heap, merged)
        return heap[0]

    def build_codes(self, node, code='', codes=None):
        if codes is None:
            codes = {}
        if node is not None:
            if node.symbol is not None:
                codes[node.symbol] = code
            self.build_codes(node.left, code + '0', codes)
            self.build_codes(node.right, code + '1', codes)
        return codes

    def encode(self, data, codes):
        encoded_data = bitarray()
        for symbol in data:
            encoded_data.extend(codes[symbol])
        return encoded_data

    def decode(self, encoded_data, root):
        current_node = root
        decoded_data = []
        for bit in encoded_data:
            if bit == 0:
                current_node = current_node.left
            else:
                current_node = current_node.right
            if current_node.symbol is not None:
                decoded_data.append(current_node.symbol)
                current_node = root
        return decoded_data

from collections import defaultdict 

def huffman_string_test():
    data = "BCAADDDCCACACAC"
    frequencies = defaultdict(int)
    for character in data: frequencies[character] += 1
    compressor = HuffmanCompressor()
    root = compressor.build_tree(frequencies)
    codes = compressor.build_codes(root)
    encoded_data = compressor.encode(data, codes)
    decoded_data = compressor.decode(encoded_data, root)
    print("String decoding successful:", data==''.join(decoded_data))

import cv2
import numpy as np

def huffman_cimage_test():
    data = cv2.imread("images/test_rgb3x3.png")
    for data_i in cv2.split(data):
        frequencies = defaultdict(int)
        for row in data_i:
            for pixel in row: frequencies[pixel] += 1
        compressor = HuffmanCompressor()
        root = compressor.build_tree(frequencies)
        codes = compressor.build_codes(root)
        encoded_data = compressor.encode(data_i.flatten(), codes)
        decoded_data = np.array(compressor.decode(encoded_data, root)).reshape(data_i.shape)
        print("Channel decoding successful:", np.array_equal(data_i, decoded_data))

if __name__ == "__main__":
    huffman_string_test()
    huffman_cimage_test()
