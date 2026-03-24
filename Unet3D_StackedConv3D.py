import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Blocco Residuo 3D (con due convoluzioni)
# ---------------------------------------------------------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2   = nn.BatchNorm3d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# ---------------------------------------------------------------------
# UNet 3D con blocchi residui
# ---------------------------------------------------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, num_levels=5, out_channels=64):
        """
        in_channels: numero di canali in ingresso (es. 3 per immagini RGB)
        base_channels: numero di canali di partenza
        num_levels: numero di livelli di down/up sampling
        out_channels: numero di canali in uscita dal blocco UNet
        """
        super().__init__()
        self.num_levels = num_levels
        
        # Iniziale blocco residuo
        self.initial_conv = ResidualBlock3D(in_channels, base_channels)
        
        # Downsampling path: ogni livello raddoppia i canali
        self.down_convs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        channels = base_channels
        for _ in range(num_levels):
            block = ResidualBlock3D(channels, channels * 2)
            self.down_convs.append(block)
            channels *= 2

        # Bottleneck
        self.bottleneck = ResidualBlock3D(channels, channels)
        
        # Upsampling path: a ogni step si dimezzano i canali e si concatenano le skip connection
        self.up_transposes = nn.ModuleList()
        self.up_convs      = nn.ModuleList()
        for _ in range(num_levels):
            self.up_transposes.append(
                nn.ConvTranspose3d(channels, channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            )
            # Dopo la concatenazione, i canali raddoppiano
            self.up_convs.append(ResidualBlock3D(channels, channels // 2))
            channels //= 2
        
        # Convoluzione finale per ottenere il numero di canali desiderato
        self.final_conv = nn.Conv3d(channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        out = self.initial_conv(x)
        
        # Downsampling
        for down in self.down_convs:
            skip_connections.append(out)
            out = self.pool(out)
            out = down(out)
        
        # Bottleneck
        out = self.bottleneck(out)
        
        # Upsampling
        for i in range(self.num_levels):
            out = self.up_transposes[i](out)
            skip = skip_connections[-(i+1)]
            # Se le dimensioni non coincidono, si effettua un padding
            if out.shape != skip.shape:
                diffT = skip.size(2) - out.size(2)
                diffH = skip.size(3) - out.size(3)
                diffW = skip.size(4) - out.size(4)
                out = F.pad(out, [diffW // 2, diffW - diffW // 2,
                                  diffH // 2, diffH - diffH // 2,
                                  diffT // 2, diffT - diffT // 2])
            out = torch.cat([out, skip], dim=1)
            out = self.up_convs[i](out)
        
        return self.final_conv(out)

# ---------------------------------------------------------------------
# Modulo StackedConv3D
# ---------------------------------------------------------------------
class GatedConv3DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv3d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        
    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c
    
    def init_hidden(self, batch_size, time_steps, height, width, device):
        h = torch.zeros(batch_size, self.hidden_dim, time_steps, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, time_steps, height, width, device=device)
        return h, c

class StackedConv3D(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128, 256,512], kernel_size=3, padding=1):
        """
        input_dim: numero di canali in ingresso per il primo layer
        hidden_dims: lista con il numero di canali per ogni layer GatedConv3DBlock
        """
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for hd in hidden_dims:
            self.layers.append(GatedConv3DBlock(current_dim, hd, kernel_size, padding))
            current_dim = hd
            
    def forward(self, x):
        batch_size, channels, time_steps, height, width = x.size()
        device = x.device
        
        h_states = []
        c_states = []
        for layer in self.layers:
            hi, ci = layer.init_hidden(batch_size, time_steps, height, width, device)
            h_states.append(hi)
            c_states.append(ci)
        
        current = x
        for i, layer in enumerate(self.layers):
            hi, ci = layer(current, h_states[i], c_states[i])
            current = hi  # l'output di uno strato diventa l'input del successivo
        
        return current


