
import cv2

class ChromaSubsamplingCompressor:
    def encode(self, channel):
        size = (channel.shape[1] // 2, channel.shape[0] // 2)
        encoded = cv2.resize(channel, size, interpolation=cv2.INTER_LINEAR)
        return encoded
    def decode(self, channel, o_shape):
        size = (o_shape[1], o_shape[0])
        decoded = cv2.resize(channel, size, interpolation=cv2.INTER_NEAREST)
        return decoded
    
from matplotlib import pyplot as plt
    
def yuv_cimage_test():
    o_data = cv2.imread("images/test_cat.png")
    data = cv2.cvtColor(o_data, cv2.COLOR_BGR2YUV)
    y_data, u_data, v_data = cv2.split(data)
    compressor = ChromaSubsamplingCompressor()
    u_encoded = compressor.encode(channel=u_data)
    v_encoded = compressor.encode(channel=v_data)
    u_decoded = compressor.decode(channel=u_encoded, o_shape=y_data.shape)
    v_decoded = compressor.decode(channel=v_encoded, o_shape=y_data.shape)
    decoded_data = cv2.merge((y_data, u_decoded, v_decoded))
    decoded_data = cv2.cvtColor(decoded_data, cv2.COLOR_YUV2BGR)
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(o_data, cv2.COLOR_BGR2RGB))
    plt.title("Original") 
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(decoded_data, cv2.COLOR_BGR2RGB))
    plt.title("Decoded")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    yuv_cimage_test()
