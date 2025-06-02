from model.unet_3_plus_model import UNet3Plus
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast


def train_net(net, device, data_path, epochs=30, batch_size=2, lr=1e-5):
    writer = SummaryWriter('runs/experiment_1')
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()




    scaler = GradScaler()
    best_loss = float('inf')
    global_step = 0

    with tqdm(total=epochs * per_epoch_num) as pbar:
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                with autocast():
                    pred = net(image)
                    loss = criterion(pred, label)

                scaler.scale(loss).backward()

                clip_grad_norm_(net.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                global_step += 1

                pbar.update(1)

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss}")

            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(net.state_dict(), 'best_model.pth')

    writer.close()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3Plus(n_channels=1, n_classes=1)
    net.to(device=device)

    data_path = "C:/Users/siche/Desktop/ddd/Brain Tumor Segmentation Dataset"
    print("It's not a program problem that the progress bar is stuck, it's just that he's calculating, so please be patient!")
    train_net(net, device, data_path, epochs=30, batch_size=2, lr=1e-5)
