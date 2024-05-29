import os
import random

import cv2
import torch
from torch.utils.data import DataLoader

from bayesopt_universal import UniversalBayesOptAttack
from dataset import MnistDataset
from submodules.pytorch_playground.mnist.model import mnist


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = mnist(pretrained=True)
    model.to(device)
    model.eval()

    # Load dataset
    train_data = MnistDataset(random.sample(range(10000), k=10))
    train_dataloader = DataLoader(
        train_data, batch_size=128, shuffle=False, drop_last=False
    )
    test_data = MnistDataset(random.sample(range(10000), k=1000))
    test_dataloader = DataLoader(
        test_data, batch_size=128, shuffle=False, drop_last=False
    )

    bayes_attack = UniversalBayesOptAttack(
        nchannel=1,
        d1=28,
        high_dim=784,
        low_dim=25,
        eps=0.3,
        num_train_images=len(train_data),
        max_iters=500,
        device=device,
    )
    delta, num_query = bayes_attack.run(model, train_dataloader)
    print("num_query: ", num_query)

    num_fool = 0
    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            x_adv = torch.clamp(x_test + delta.squeeze(-1), min=0.0, max=1.0)
            x_adv = x_adv.to(device)
            pred = torch.argmax(model(x_adv.view(x_adv.shape[0], -1)), dim=1)
            num_fool += torch.count_nonzero(pred.cpu() != y_test).item()
    print("fooling rate: ", num_fool, "/", len(test_data))

    os.makedirs("outputs", exist_ok=True)
    x_adv = x_adv.squeeze().cpu().numpy()
    for i in range(min(10, x_adv.shape[0])):
        cv2.imwrite(f"outputs/{i}.png", (x_adv[i] * 255).astype("uint8"))
