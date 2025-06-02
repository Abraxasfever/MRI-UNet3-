import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.eval()
    tests_path = glob.glob('images/111/*.jpg')
    for test_path in tests_path:
        save_res_path = test_path.split('.')[0] + '_res.png'

        img = cv2.imread(test_path)
        origin_shape = img.shape
        print(origin_shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.3] = 255
        pred[pred < 0.3] = 0
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_res_path, pred)