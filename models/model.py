#  Copyright (c) 2023. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

from torch import nn

from models.decoder import RGBDecoder, ThermalDecoder
from models.encoder import RGBEncoder, ThermalEncoder
from models.vector_quantizer import VectorQuantizer


class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._rgb_encoder = RGBEncoder(3, num_hiddens,
                                       num_residual_layers,
                                       num_residual_hiddens)
        self._thermal_encoder = ThermalEncoder(1, num_hiddens,
                                               num_residual_layers,
                                               num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self._vq_vae = VectorQuantizer(num_embeddings,
                                       embedding_dim,
                                       commitment_cost)

        self._rgb_decoder = RGBDecoder(embedding_dim,
                                       num_hiddens,
                                       num_residual_layers,
                                       num_residual_hiddens)

        self._thermal_decoder = ThermalDecoder(embedding_dim,
                                               num_hiddens,
                                               num_residual_layers,
                                               num_residual_hiddens)

    def forward(self, x):
        z_rgb = self._rgb_encoder(x)
        z_thermal = self._thermal_encoder(x)

        z = z_rgb + z_thermal

        z = self._pre_vq_conv(z)

        loss, quantized, perplexity, _ = self._vq_vae(z)

        x_rgb_recon = self._rgb_decoder(quantized)
        x_thermal_recon = self._thermal_decoder(quantized)

        return loss, x_rgb_recon, x_thermal_recon, perplexity
