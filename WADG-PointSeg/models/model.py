import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torch.nn.functional as F
from torch.nn import Parameter

__all__ = ['WADGPointSegNet']
class WADGPointSegNet(nn.Module):
    def __init__(self, **kwargs):
        super(WADGPointSegNet, self).__init__()

        cr = kwargs.get('cr', 1.0)  # Retrieve channel scaling factor from kwargs
        channels = [int(cr * x) for x in [32, 32, 64, 128, 256, 256, 128, 96, 96]]
        self.use_upsample = kwargs.get('use_upsample', True)  # Allow for optional upsample usage

        # Define initial layers
        self.stem = nn.Sequential(
            spnn.Conv3d(4, channels[0], kernel_size=3, stride=1),
            spnn.BatchNorm(channels[0]), spnn.ReLU(inplace=True),
            spnn.Conv3d(channels[0], channels[0], kernel_size=3, stride=1),
            spnn.BatchNorm(channels[0]), spnn.ReLU(inplace=True)
        )

        # Define downsampling stages
        self.stage1 = self._make_stage(channels[0], channels[1], stride=2)
        self.stage2 = self._make_stage(channels[1], channels[2], stride=2)
        self.stage3 = self._make_stage(channels[2], channels[3], stride=2)
        self.stage4 = self._make_stage(channels[3], channels[4], stride=2)

        # Define upsampling stages
        self.up1 = self._make_upsample_block(channels[4], channels[5], channels[3])
        self.up2 = self._make_upsample_block(channels[5], channels[6], channels[2])
        self.up3 = self._make_upsample_block(channels[6], channels[7], channels[1])
        self.up4 = self._make_upsample_block(channels[7], channels[8], channels[0])

        # Define classifiers
        self.classifier = nn.Linear(channels[8], num_classes)
        self.classifier_aux = nn.Linear(128, num_classes)

        # Define projection layers
        self.projection = self._make_projection_layer(channels[8], 128)
        self.projection_aux1 = self._make_projection_layer(64, 64)
        self.projection_aux2 = self._make_projection_layer(32, 16)

        # Memory bank and feature scaling
        self.momentum = 0.99
        self.register_buffer("memory_bank", torch.randn(num_classes, 128) * 0)
        # Initialize feature scaling module to adaptively scale feature representations
        self.feature_scaling = AdaptiveFeatureScaling(num_channels=128)
        # Initialize pseudo label generator for generating high-confidence pseudo labels
        self.pseudo_label_generator = PseudoLabelGenerator(128, 64, num_classes)
        # Define MultiAttentionFusion module
        self.attention_fusion = MultiAttentionFusion(feature_dim=128, device='cuda')
        # Initialize weights
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            BasicConvolutionBlock(in_channels, out_channels, kernel_size=2, stride=stride),
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def _make_upsample_block(self, in_channels, out_channels, skip_channels):
        return nn.ModuleList([
            BasicDeconvolutionBlock(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Sequential(
                ResidualBlock(out_channels + skip_channels, out_channels),
                ResidualBlock(out_channels, out_channels)
            )
        ])

    def _make_projection_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels)
        )

    @torch.no_grad()
    def update_memory_bank(self, feature, initialize=False):
        if initialize:
            self.memory_bank = feature
        else:
            self.memory_bank = self.memory_bank * self.momentum + feature * (1 - self.momentum)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Downsampling
        features = [self.stem(x)]
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            features.append(stage(features[-1]))

        # Reverse features for upsampling
        features.reverse()
        upsample_output = features[0]
        
        # Upsampling
        for i, upsample_block in enumerate([self.up1, self.up2, self.up3, self.up4]):
            upsample_output = upsample_block[0](upsample_output)  # Upsample layer
            upsample_output = torchsparse.cat([upsample_output, features[i + 1]])  # Concatenate with skip connection
            upsample_output = upsample_block[1](upsample_output)  # Further processing

        final_feature_map = upsample_output

        # Main output (final classification)
        minkowski_net_output = self.classifier(final_feature_map.F)

        # Projection and scaling for auxiliary task
        projected_features = self.projection(final_feature_map.F)
        scaled_features = self.feature_scaling(projected_features)

        # Applying attention fusion for auxiliary output
        fused_features = self.attention_fusion(scaled_features)

        #Using the auxiliary classifier to generate pseudo labels from the fused features
        pseudo_label_output = self.classifier_aux(fused_features)

        return minkowski_net_output, fused_features, pseudo_label_output


class BasicConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(BasicConvolutionBlock, self).__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicDeconvolutionBlock, self).__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, transposed=True),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=1),
            spnn.BatchNorm(out_channels),
        )
        self.downsample = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            spnn.BatchNorm(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Sequential()

        self.relu = spnn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.net(x)
        return self.relu(out + residual)


class AdaptiveFeatureScaling(nn.Module):
    def __init__(self, num_channels):
        super(AdaptiveFeatureScaling, self).__init__()

        # Initialize gamma and beta parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(1, num_channels))  # Scaling factor
        self.beta = nn.Parameter(torch.zeros(1, num_channels))  # Shifting factor

        # Define the scale predictor network
        self.scale_predictor = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, num_channels),
        )

        # Define the shift predictor network
        self.shift_predictor = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // 2, num_channels)
        )

    def forward(self, x):
        # Compute the mean of input features
        feature_mean = x.mean(dim=0, keepdim=True)

        # Normalize the input features using Layer Normalization
        normalized_features = F.layer_norm(x, x.size()[1:])

        # Predict the scale and shift factors
        scale_factor = 1 + self.scale_predictor(feature_mean).sigmoid() * self.gamma
        shift_factor = self.shift_predictor(feature_mean) * self.beta

        # Apply the adaptive scaling and shifting to the normalized features
        scaled_shifted_features = normalized_features * scale_factor + shift_factor

        return scaled_shifted_features



class SampledSelfAttention(nn.Module):
    """
    Applies self-attention on a randomly sampled subset of features.
    This module reduces the computational complexity of traditional self-attention
    by performing attention calculations on a subset of the input features, thereby
    making it more suitable for large-scale input data.
    """

    def __init__(self, feature_dim, num_samples=1024):
        """
        Initializes the SampledSelfAttention module.

        Args:
            feature_dim (int): The dimensionality of the input features (d).
            num_samples (int): The number of features to sample for attention (n).
        """
        super(SampledSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, features):
        """
        Forward pass to compute self-attention on a sampled subset of features.

        Args:
            features (Tensor): Input feature tensor of shape (N, feature_dim),
                               where N is the number of features (points).

        Returns:
            Tensor: Aggregated output tensor of shape (N, feature_dim), where each feature
                    incorporates information from a sampled subset via attention.
                    This output effectively balances computational efficiency and
                    information preservation by applying global attention values uniformly
                    across the input.
        """
        num_features, _ = features.shape
        
        # Randomly sample a subset of features for attention calculation to reduce complexity
        sampled_indices = torch.randint(0, num_features, (self.num_samples,))
        sampled_features = features[sampled_indices].to('cuda')

        # Compute query, key, value projections using learnable linear transformations
        queries = self.query(sampled_features)
        keys = self.key(sampled_features)
        values = self.value(sampled_features)

        # Calculate attention weights using scaled dot-product attention
        # softmax is applied after scaling attention scores to maintain numerical stability
        attention_scores = queries @ keys.transpose(-2, -1)
        max_attention_score = torch.max(attention_scores)  # For numerical stability
        attention_weights = F.softmax(attention_scores / max_attention_score, dim=-1)

        # Weighted sum of value vectors based on attention weights
        attention_output = attention_weights @ values

        # Compute the mean of the attention output and replicate it to match the original input size
        aggregated_output = attention_output.mean(dim=0, keepdim=True).expand(num_features, -1)

        return aggregated_output

class BlockSelfAttention(nn.Module):
    """
    Applies non-parametric self-attention using distance-based Gaussian weighting within blocks of features.
    """
    def __init__(self, feature_dim, block_size=1024):
        """
        Initializes the BlockSelfAttention module.

        Args:
            feature_dim (int): The dimensionality of the input features.
            block_size (int): The size of each block for applying attention.
        """
        super(BlockSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.block_size = block_size

    def forward(self, features):
        """
        Forward pass to compute self-attention within blocks using Gaussian weighting.

        Args:
            features (Tensor): Input feature tensor of shape (N, feature_dim),
                               where N is the number of features.

        Returns:
            Tensor: Attention-enhanced feature tensor of shape (N, feature_dim),
                    with local attention applied within each block.
        """
        num_features, _ = features.shape
        num_blocks = num_features // self.block_size

        # Output tensor initialization
        attention_output = torch.zeros_like(features).to(features.device)

        # Process each block independently
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            block = features[start_idx:end_idx]

            # Compute pairwise Euclidean distances and attention weights
            distances = torch.cdist(block, block, p=2).pow(2)
            attention_weights = torch.exp(-distances / (2 * (self.feature_dim / 10.0)))

            # Apply attention weights to block
            block_attention_output = attention_weights @ block

            # Normalize and store the output
            attention_output[start_idx:end_idx] = block_attention_output / self.block_size

        return attention_output

class MultiAttentionFusion(nn.Module):
    """
    Combines outputs from sampled self-attention and block-based Gaussian self-attention.
    """
    def __init__(self, feature_dim, num_samples=1024, device='cuda'):
        """
        Initializes the MultiAttentionFusion module.

        Args:
            feature_dim (int): The dimensionality of the input features.
            num_samples (int): The number of features to sample for sampled self-attention.
            device (str): Device to run the attention mechanisms on ('cuda' or 'cpu').
        """
        super(MultiAttentionFusion, self).__init__()
        self.sampled_attention = SampledSelfAttention(feature_dim, num_samples).to(device)
        self.gaussian_attention = BlockSelfAttention(feature_dim).to(device)
        self.attention_weights = nn.Parameter(torch.tensor([1.0, 1.0])).to(device)

    def forward(self, features):
        """
        Forward pass to compute combined attention from different mechanisms.

        Args:
            features (Tensor): Input feature tensor of shape (N, feature_dim),
                               where N is the number of features.

        Returns:
            Tensor: Weighted combination of attention outputs, shape (N, feature_dim).
        """
        # Compute outputs from both attention mechanisms
        sampled_attention_output = self.sampled_attention(features)
        gaussian_attention_output = self.gaussian_attention(features)

        # Slight noise addition for numerical stability
        noise = torch.randn(self.attention_weights.size()) * 1e-6

        # Compute softmax weights with max subtraction for numerical stability
        max_weight = torch.max(self.attention_weights)
        normalized_weights = F.softmax((self.attention_weights - max_weight) / 10.0 + noise.to(self.attention_weights.device), dim=0)

        # Weighted sum of the attention outputs
        combined_output = normalized_weights[0] * sampled_attention_output + normalized_weights[1] * gaussian_attention_output

        return combined_output


class PseudoLabelGenerator(nn.Module):
    """
    Generates pseudo labels from input features using a simple feedforward neural network.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the PseudoLabelGenerator.

        Args:
            input_dim (int): Number of input channels/features.
            output_dim (int): Number of output classes for pseudo labels.
            hidden_dim (int): Number of hidden units in the intermediate layer.
        """
        super(PseudoLabelGenerator, self).__init__()
        
        # Define the feedforward network for pseudo label generation
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input to hidden layer
            nn.ReLU(inplace=True),             # Activation function
            nn.Linear(hidden_dim, output_dim), # Hidden to output layer
            nn.Softmax(dim=-1)                 # Softmax for pseudo label probabilities
        )

    def forward(self, features):
        """
        Forward pass to generate pseudo labels.

        Args:
            features (Tensor): Input feature tensor.

        Returns:
            Tensor: Pseudo label probabilities.
        """
        return self.network(features)
