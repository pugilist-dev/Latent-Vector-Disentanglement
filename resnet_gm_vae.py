import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip_connection(x)

        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.elu(out)

        return out

class Encoder(nn.Module):
    def __init__(self, z_dim=8, K=8, dropout=0.3):
        super(Encoder, self).__init__()
        self.K = K
        # Define the initial convolutional layers and ResNet blocks as before
        self.layers = nn.Sequential(
            ResNetBlock(4, 64),
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 256, stride=2),
            #ResNetBlock(256, 512, stride=2),
            #ResNetBlock(512, 512, stride=2),
        )
        self.dropout = nn.Dropout(p=dropout)
        # Assuming the output feature map size is [512, 2, 2] for an input of [3, 224, 224]
        # Adjust the size (512 * 2 * 2) accordingly if your feature map size is different
        self.fc_mu = nn.Linear(256 * 19 * 19, z_dim * K)
        self.fc_logvar = nn.Linear(256 * 19 * 19, z_dim * K)
        self.fc_pi = nn.Linear(256 * 19 * 19, K)

    def forward(self, x):
        x = self.layers(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        pi_logits = self.fc_pi(x)
        return mu, logvar, pi_logits
    
class Decoder(nn.Module):
    def __init__(self, z_dim=8):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(z_dim, 256 * 19 * 19)
        
        self.layers = nn.Sequential(
            #nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            #ResNetBlock(512, 512),
            #nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            #ResNetBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ResNetBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ResNetBlock(64, 64),
            nn.Conv2d(64, 4, kernel_size=3, padding=1),  # Assuming 4 channel output
        )
        

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 19, 19)
        x = self.layers(x)
        x = torch.sigmoid(x)
        # Use interpolation to adjust the final size precisely
        x = F.interpolate(x, size=(75, 75), mode='bilinear', align_corners=False)
        return x

class deanGMVAE(nn.Module):
    
    def __init__(self, z_dim=8, beta=5, K=7, dropout=0.2):
        super(deanGMVAE, self).__init__()
        self.latent_size = z_dim
        self.K = K
        self.beta = beta
        self.dropout = dropout
        self.encoder = Encoder(z_dim=self.latent_size,K=self.K, dropout=self.dropout)
        self.decoder = Decoder(z_dim=self.latent_size)
        self.register_buffer('pi_prior', torch.full((K,), fill_value=1.0/K))
        
    def forward(self, x, temperature=0.5):
        mean, logvar, pi_logits = self.encoder(x)
        y = deanGMVAE.sample_concrete(pi_logits, temperature)
        z = deanGMVAE.sample(mean, logvar, y, self.K, self.latent_size)  # Note the additional arguments here
        z_prime = self.attention_grab(z)
        recon_x = self.decoder(z_prime)
        return recon_x, mean, logvar, pi_logits, z
    
    def pairwise_cosine_similarity(self, x):
        x_normalized = x / x.norm(dim=1, keepdim=True)
        similarity_matrix = torch.mm(x_normalized, x_normalized.t())
        return similarity_matrix
    
    def attention_grab(self, z):
        cosine_sim = self.pairwise_cosine_similarity(z)
        cosine_sim.fill_diagonal_(1)
        z_prime = z.new_empty(z.size()).to(z.device)
        #z_prime = torch.mm(cosine_sim, z)
        for i in cosine_sim.shape[0]:
            z_prime[i,:] = torch.sum(cosine_sim[i,:]*z, dim=0)
        return z_prime
    
    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -torch.log(-torch.log(U+eps)+eps)
    
    @staticmethod
    def sample_concrete(logits, temperature):
        gumbel_noise = deanGMVAE.sample_gumbel(logits.size())
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)
    
    @staticmethod
    def sample(mean, logvar, y, K, latent_size):
        batch_size = mean.size(0)
        
        # Reshape mean and logvar to [batch_size, K, latent_size] to separate components
        mean = mean.view(batch_size, K, latent_size)
        logvar = logvar.view(batch_size, K, latent_size)
        
        # Compute standard deviation
        std = torch.exp(0.5 * logvar)
        
        # Sample epsilon for each Gaussian component
        eps = torch.randn_like(std)
        
        # Reparameterize each component
        z_components = mean + eps * std  # Shape: [batch_size, K, latent_size]
        
        # Weight samples by responsibilities y
        # First, ensure y is correctly shaped for weighting the components
        y = y.unsqueeze(-1)  # Shape: [batch_size, K, 1] to broadcast over latent_size
        z_weighted = torch.sum(z_components * y, dim=1)  # Shape: [batch_size, latent_size]
        
        return z_weighted
    
    '''
    def compute_kl_concrete(self,logits, pi_prior, temperature, wt):
        q_y = F.softmax(logits / temperature, dim=-1)
        log_q_y = torch.log(q_y + 1e-20)  # Adding a small constant to prevent log(0)
        log_pi_prior = torch.log(pi_prior + 1e-20)
        kl_diverge_y = torch.sum(q_y * (log_q_y - log_pi_prior), dim=-1).mean()
        return kl_diverge_y*wt
    '''
        
    def loss(self, recon_x, x, mu, pi_logits, temperature, logvar, current_beta):
        l1_loss = self.calculate_l1_loss(recon_x, x)
        kl_loss = current_beta*self.calculate_gaussian_kl_loss(mu, logvar)
        cat_loss = self.compute_categorical_loss(pi_logits, temperature)
        total_loss = l1_loss + kl_loss + cat_loss
        return total_loss, l1_loss, kl_loss, cat_loss
    
    def calculate_l1_loss(self, recon_x, x):
        batch_size = x.size(0)
        return F.l1_loss(recon_x, x, reduction="sum") / batch_size

    def calculate_gaussian_kl_loss(self, mu, logvar):
        batch_size = mu.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        return kl_div

    def compute_categorical_loss(self, pi_logits, temperature=1.0):
        batch_size, num_classes = pi_logits.shape
        
        # Define the target distribution as uniform across the 8 classes
        targets = torch.full_like(pi_logits, fill_value=1.0/num_classes)
        
        # Apply temperature scaling on logits and compute the log softmax
        log_q = F.log_softmax(pi_logits / temperature, dim=-1)
        
        # Compute the categorical loss
        categorical_loss = -torch.mean(torch.sum(targets * log_q, dim=-1))
        
        return categorical_loss
