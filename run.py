from midas_depth import MidasDepth
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    depth_predictor = MidasDepth()
    img_path = r'example/0.jpg'
    depth = depth_predictor.predict(cv2.imread(depth_predictor))
    plt.imshow(depth)
    plt.show()