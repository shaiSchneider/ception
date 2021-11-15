from midas_depth import MidasDepth
from sky_detector import SkyDetector
import cv2
import os
import tqdm
import numpy as np

if __name__ == "__main__":
    eps = 1e-3
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Ground", cv2.WINDOW_NORMAL)
    depth_predictor = MidasDepth()
    sky_predictor = SkyDetector()
    base_dir = r"/media/shai/New Volume/ception_data/Ception_12072021_P1/LB5_IMAGES/LB5, Camera -30deg, 0"
    filenames = os.listdir(base_dir)
    for file in tqdm.tqdm(filenames):
        img_path = os.path.join(base_dir, file)
        img = cv2.imread(img_path)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = depth_predictor.predict(rgb.copy())
        depth = 1./(depth + eps)
        depth[depth > 0.8 * depth.max()] = 0
        depth[np.isnan(depth)] = 0
        sky_mask = sky_predictor.predict(rgb.copy())
        ground_mask = sky_predictor.get_ground_mask(sky_mask)
        depth = depth * ground_mask
        cv2.imshow("RGB", img)
        cv2.imshow("Depth", depth)
        cv2.imshow("Ground", ground_mask)
        if cv2.waitKey(1) & 0xFF == 27:
                break 