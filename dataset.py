import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


DATA_PATH = "./datasets/"


class MnistDataset(Dataset):
    def __init__(self, indices_to_use=None) -> None:
        super().__init__()
        data = MNIST(root=DATA_PATH, train=False, download=True)
        arrays = (data.data.numpy(), data.targets.numpy())
        if indices_to_use is not None:
            arrays = tuple([array[indices_to_use] for array in arrays])

        self.data = arrays[0]
        self.targets = arrays[1]

    def __getitem__(self, index: int):
        return F.to_tensor(self.data[index]), self.targets[index]

    def __len__(self):
        return len(self.data)
