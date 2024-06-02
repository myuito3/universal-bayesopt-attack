import argparse
import os
import random

import cv2
import torch
from torch.utils.data import DataLoader

from bayesopt_universal import UniversalBayesOptAttack
from dataset import MnistDataset, Cifar10Dataset
from submodules.pytorch_playground.mnist.model import mnist
from submodules.pytorch_playground.cifar.model import cifar10


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and dataset
    if args.model == "mnist":
        model = mnist(pretrained=True)
        input_shape = (784,)
        train_data = MnistDataset(random.sample(range(10000), k=args.num_train_images))
        test_data = MnistDataset(random.sample(range(10000), k=args.num_test_images))
    elif args.model == "cifar10":
        model = cifar10(n_channel=128, pretrained=True)
        input_shape = (3, 32, 32)
        train_data = Cifar10Dataset(
            random.sample(range(10000), k=args.num_train_images)
        )
        test_data = Cifar10Dataset(random.sample(range(10000), k=args.num_test_images))
    model.to(device)
    model.eval()

    train_dataloader = DataLoader(
        train_data, batch_size=128, shuffle=False, drop_last=False
    )
    test_dataloader = DataLoader(
        test_data, batch_size=128, shuffle=False, drop_last=False
    )

    # Execute attack
    bayes_attack = UniversalBayesOptAttack(
        nchannel=args.nchannel,
        d1=args.d1,
        high_dim=args.high_dim,
        low_dim=args.low_dim,
        eps=args.eps,
        input_shape=input_shape,
        num_train_images=len(train_data),
        setting=args.setting,
        max_iters=args.max_iters,
        dim_reduction=args.dim_reduction,
        device=device,
    )
    delta, num_query = bayes_attack.run(model, train_dataloader)
    print("num_query: ", num_query)

    # Evaluation
    num_fool = 0
    with torch.no_grad():
        delta = delta.squeeze()[None, ...]
        for x_test, y_test in test_dataloader:
            x_adv = torch.clamp(x_test + delta, min=0.0, max=1.0)
            x_adv = x_adv.float().to(device)
            pred = torch.argmax(model(x_adv.view(x_adv.shape[0], *input_shape)), dim=1)
            num_fool += torch.count_nonzero(pred.cpu() != y_test).item()
    print("fooling rate: ", num_fool, "/", len(test_data))

    os.makedirs("outputs", exist_ok=True)
    x_adv = x_adv.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    for i in range(min(10, x_adv.shape[0])):
        cv2.imwrite(f"outputs/{i}.png", (x_adv[i] * 255).astype("uint8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["mnist", "cifar10"],
        default="mnist",
        help="model",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["score", "decision"],
        default="score",
        help="setting",
    )
    parser.add_argument("--max_iters", type=int, default=2000, help="max iters")
    parser.add_argument(
        "--num_train_images", type=int, default=10, help="num train images"
    )
    parser.add_argument(
        "--num_test_images", type=int, default=1000, help="num test images"
    )

    args = parser.parse_args()

    if args.model == "mnist":
        args.nchannel = 1
        args.d1 = 28
        args.high_dim = args.d1 * args.d1
        args.low_dim = 25
        args.eps = 0.3
        args.dim_reduction = "NN"
    elif args.model == "cifar10":
        args.nchannel = 3
        args.d1 = 32
        args.high_dim = args.d1 * args.d1
        args.low_dim = 16
        args.eps = 0.05
        args.dim_reduction = "Tile"

    main(args)
