from datasets.FLIRDataset import FLIRDataset
from torch.utils.data import DataLoader


def train():
    # 1. Load the dataset
    flir_dataset = FLIRDataset(root_dir='data', train=True)

    # 2. Create the dataloaders
    flir_dataloader = DataLoader(flir_dataset, batch_size=4, shuffle=True)

    # 3. Create the model
    # 4. Train the model

    # Iterate over the dataloader
    for i, data in enumerate(flir_dataloader):
        print(i, data)

    # 5. Evaluate the model
    # 6. Save the model


if __name__ == '__main__':
    train()
