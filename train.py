import numpy as np
from six.moves import xrange
import torch
from datasets.FLIRDataset import FLIRDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.mvqvae import MultimodalVQVAE

batch_size = 8
num_training_updates = 10000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

learning_rate = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    # 1. Load the dataset
    flir_dataset = FLIRDataset(root_dir='data', train=True)

    # 2. Create the dataloaders
    flir_dataloader = DataLoader(flir_dataset, batch_size=batch_size, shuffle=True)

    # 3. Create the model
    model = MultimodalVQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                            num_embeddings, embedding_dim, commitment_cost).to(device)

    # 4. Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in xrange(num_training_updates):
        rgb_img, thermal_img = next(iter(flir_dataloader))
        rgb_img = rgb_img.to(device)
        thermal_img = thermal_img.to(device)
        optimizer.zero_grad()

        vq_loss, rgb_recon, thermal_recon, perplexity = model([rgb_img, thermal_img])

        # 5. Evaluate the model
        rgb_recon_error = F.mse_loss(rgb_recon, rgb_img)
        thermal_recon_error = F.mse_loss(thermal_recon, thermal_img)

        recon_error = rgb_recon_error + thermal_recon_error

        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()

        # 6. Save the model
        if (i + 1) % 1000 == 0:
            torch.save(model.state_dict(), 'checkpoints/model_%d.pth' % (i + 1))


if __name__ == '__main__':
    train()
