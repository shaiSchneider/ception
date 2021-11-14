import torch
import cv2
import matplotlib.pyplot as plt

class MidasDepth:
    def __init__(self, model_type="DPT_Large"):
        self.model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)
        self.midas.to(self.device)
        self.midas.eval()

        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

    def predict(self, img):

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
    
        return output

if __name__ == "__main__":
    depth_predictor = MidasDepth()
    img_path = r'example/0.jpg'
    depth = depth_predictor.predict(cv2.imread(img_path))
    cv2.imshow("",depth)
    cv2.waitKey(0)