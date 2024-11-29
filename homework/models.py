# ChatGPT was used to help define the models
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 2 * n_track * 2
        hidden_dim = 128
        output_dim = 2 * n_waypoints

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.
        """
        x = torch.cat((track_left.flatten(start_dim=1), track_right.flatten(start_dim=1)), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = x.view(-1, self.n_waypoints, 2)
        
        return x

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        self.track_embed = nn.Linear(2, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)

        self.output_layer = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)
        """
        batch_size = track_left.size(0)

        track_left_embedded = self.track_embed(track_left)
        track_right_embedded = self.track_embed(track_right)

        track_embedded = torch.cat((track_left_embedded, track_right_embedded), dim=1)

        pos_enc = torch.arange(0, 2 * self.n_track, device=track_embedded.device).unsqueeze(0).repeat(batch_size, 1)
        pos_enc = pos_enc.unsqueeze(-1).float()
        pos_enc = pos_enc.expand(-1, -1, self.d_model)

        track_embedded += pos_enc

        query_embedded = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        track_embedded = self.transformer_encoder(track_embedded.permute(1, 0, 2))

        transformer_output = self.transformer_decoder(
            tgt=query_embedded,
            memory=track_embedded,
        )

        waypoints = self.output_layer(transformer_output.permute(1, 0, 2))

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()

        self.n_waypoints = n_waypoints
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass to predict waypoints from the image.
        """
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        waypoints = x.view(-1, self.n_waypoints, 2)

        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
