import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np

# CelebA (VAE)
# Input 64x64x3.
# Adam 1e-4
# Encoder Conv 32x4x4 (stride 2), 32x4x4 (stride 2), 64x4x4 (stride 2),
# 64x4x4 (stride 2), FC 256. ReLU activation.
# Latents 32
# Decoder Deconv reverse of encoder. ReLU activation. Gaussian.

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
class DARTS(nn.Module):
    def __init__(self, num_channels=8, n_nodes=5):
        super(DARTS, self).__init__()
        self.e = {}
        for i in range(1, n_nodes+1):
            for j in range(i+1, n_nodes+1):
                self.e[str(i)+","+str(j)] = conv(3,8) if i == 1 else conv(8,8)
        # self.e12 = conv(3, 8)
        # self.e13 = conv(3, 8)
        # self.e14 = conv(3, 8)
        # self.e15 = conv(3, 8)
        # self.e16 = conv(3, 8)
        # self.e17 = conv(3, 8)
        # self.e23 = conv(8, 8)
        # self.e24 = conv(8, 8)
        # self.e25 = conv(8, 8)
        # self.e26 = conv(8, 8)
        # self.e27 = conv(8, 8)
        # self.e34 = conv(8, 8)
        # self.e35 = conv(8, 8)
        # self.e36 = conv(8, 8)
        # self.e37 = conv(8, 8)
        # self.e45 = conv(8, 8)
        # self.e46 = conv(8, 8)
        # self.e47 = conv(8, 8)
        # self.e56 = conv(8, 8)
        # self.e57 = conv(8, 8)
        # self.e67 = conv(8, 8)

    def forward(self, x, n_nodes=5):
        e = {}
        for i in range(1, n_nodes+1):
            e[str(i)+",0"]=0
            for k in range(1, i):
                e[str(i)+",0"]+=e[str(k)+","+str(i)]
            for j in range(i+1, n_nodes+1):
                e[str(i)+","+str(j)] = self.e[str(i)+","+str(j)](x) if i == 1 else self.e[str(i)+","+str(j)](e[str(i)+",0"])
        # e12 = self.e12(x)
        # e13 = self.e13(x)
        # e14 = self.e14(x)
        # e15 = self.e15(x)
        # e16 = self.e16(x)
        # e17 = self.e17(x)
        # e23 = self.e23(e12)
        # e24 = self.e24(e12)
        # e25 = self.e25(e12)
        # e26 = self.e26(e12)
        # e27 = self.e27(e12)
        # e3 = sum([e13, e23])
        # e34 = self.e34(e3)
        # e35 = self.e35(e3)
        # e36 = self.e36(e3)
        # e37 = self.e37(e3)
        # e4 = sum([e14, e24, e34])
        # e45 = self.e45(e4)
        # e46 = self.e46(e4)
        # e47 = self.e47(e4)
        # e5 = sum([e15, e25, e35, e45])
        # e56 = self.e56(e5)
        # e57 = self.e57(e5)
        # e6 = sum([e16, e26, e36, e46, e56])
        # e67 = self.e67(e6)
        # e7 = torch.cat((e17, e27, e37, e47, e57, e67), dim=1)
        cat_op = []
        for i in range(1, n_nodes):
            cat_op.append(e[str(i)+","+str(n_nodes)])
        return torch.cat(tuple(cat_op), dim=1)

class Level1Op(nn.Module):
    def __init__(self, channels_in):
        super(Level1Op, self).__init__()
        if channels_in == 3:
            channels_2 = 8
        else:
            channels_2 = channels_in
        self.e12 = conv(channels_in, channels_2)
        self.e13 = conv(channels_in, channels_2)
        self.e23 = conv(channels_2, channels_2)

    def forward(self, x):
        e12 = self.e12(x)
        e13 = self.e13(x)
        e23 = self.e23(e12)
        return sum([e13, e23])

class HDARTS(nn.Module):
    def __init__(self, channels_in, n_nodes=5):
        super(HDARTS, self).__init__()
        self.e = {}
        for i in range(1, n_nodes+1):
            for j in range(i+1, n_nodes+1):
                self.e[str(i)+","+str(j)] = Level1Op(channels_in) if i==1 else Level1Op(8)
        # self.e12 = Level1Op(channels_in)
        # self.e13 = Level1Op(channels_in)
        # self.e23 = Level1Op(channels_in)

    def forward(self, x, n_nodes=5):
        e = {}
        for i in range(1, n_nodes+1):
            e[str(i)+",0"]=0
            for k in range(1, i):
                e[str(i)+",0"]+=e[str(k)+","+str(i)]
            for j in range(i+1, n_nodes+1):
                e[str(i)+","+str(j)] = self.e[str(i)+","+str(j)](x) if i == 1 else self.e[str(i)+","+str(j)](e[str(i)+",0"])
        # e12 = self.e12(x)
        # e13 = self.e13(x)
        # e23 = self.e23(e12)
        cat_op = []
        for i in range(1, n_nodes):
            cat_op.append(e[str(i)+","+str(n_nodes)])
        return torch.cat(tuple(cat_op), dim=1)

class Decoder(nn.Module):
    def __init__(self, n_nodes=5):
        super(Decoder, self).__init__()
        #added self.l1 here
#        self.l1 = conv(80, 64)
        self.l1 = conv((n_nodes-1)*8, 16)
        self.l2 = conv(16, 8)
        self.l3 = conv(8, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        
        x = self.l3(x)
        #x = self.l5(x)
        
        return self.sigmoid(x)

class BetaVAE(nn.Module):

    def __init__(self, latent_size=32, beta=1):
        super(BetaVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.encoder = DARTS() #TODO: This is the line to change to select the DARTS or HDARTS Model
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)

        # decoder
        self.fc_z = nn.Linear(latent_size, 256)
        self.decoder = Decoder()
        
    def encode(self, x):
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = x.view(-1, 256)
        #print(x.shape)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z, n_nodes=5):
        z = self.fc_z(z)
#        z = z.view(-1, 80, 64, 64)
        #z = z.view(-1, 128, 128, 128)
        z = z.view(-1, (n_nodes-1)*8, 64, 64)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        #print(recon_x.shape)
        #print(x.shape)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = (recon_loss + self.beta * kl_diverge) / x.shape[0]
        entanglement = kl_diverge / x.shape[0]
        return total_loss, entanglement  # divide total loss by batch size

    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        utils.restore(self, file_path)

    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)
