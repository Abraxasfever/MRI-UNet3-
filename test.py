import os
from tqdm import tqdm
import numpy as np
import torch
import cv2
from model.unet_3_plus_model_Squeeze import UNet3Plus
from utils.utils_metrics import compute_mIoU, show_results


def cal_miou(test_dir="C:/Users/siche/Desktop/ddd/Brain Tumor Segmentation Dataset/Test_Images",
             pred_dir="C:/Users/siche/Desktop/ddd/Brain Tumor Segmentation Dataset/results",
             gt_dir="C:/Users/siche/Desktop/ddd/Brain Tumor Segmentation Dataset/Test_Labels"):
    miou_mode = 0
    num_classes = 2
    name_classes = ["background", "neoplasms"]

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet3Plus(n_channels=1, n_classes=1)
        net.to(device=device)
        missing_keys, unexpected_keys = net.load_state_dict(torch.load('best_model.pth', map_location=device),
                                                            strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            img = cv2.imread(image_path)
            origin_shape = img.shape

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (512, 512))

            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = net(img_tensor)

            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy().squeeze()
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0

            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")

        miou_out_path = "results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == '__main__':
    cal_miou()
