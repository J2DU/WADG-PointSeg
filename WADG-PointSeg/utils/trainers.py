import os
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
import torch.nn.functional as F

import time
from typing import Any, Dict, List, Optional, Callable
from torch.utils.data import DataLoader

from torchpack.callbacks import Callback, Callbacks
from torchpack.train.exception import StopTraining
from torchpack.train.summary import Summary
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from torchpack.utils.config import configs
from .callbacks import MeanIoU

import tqdm

__all__ = ['WADGPointSegNetTrainer']

class WADGPointSegNetTrainer(Trainer):
    """
    Trainer class for WADGPointSegNet model, managing training loop, evaluation, and checkpoints.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False,
                 lambda_param: float = 0.1:
        """
        Initialize the WADGPointSegNetTrainer class.

        Args:
            model (nn.Module): The neural network model to train.
            criterion (Callable): The loss function.
            optimizer (Optimizer): The optimizer for training.
            scheduler (Scheduler): The learning rate scheduler.
            num_workers (int): Number of workers for data loading.
            seed (int): Random seed for reproducibility.
            amp_enabled (bool): Flag to enable automatic mixed precision.
            lambda_param (float): Regularization parameter for losses.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1
        self.lambda_param = lambda_param

        # Initialize Mutual Information Neural Estimator (MINE)
        self.mine_net = MINE(input_size=16).to("cuda")
        self.mine_optimizer = torch.optim.Adam(self.mine_net.parameters(), lr=0.01)

    def _before_epoch(self) -> None:
        """
        Set up necessary configurations before each epoch.
        """
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        np.random.seed(self.seed + (self.epoch_num - 1) * self.num_workers)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a training step.

        Args:
            feed_dict (Dict[str, Any]): Dictionary containing input data.

        Returns:
            Dict[str, Any]: Output predictions and corresponding targets.
        """
        # Prepare inputs
        inputs = {k: v.cuda() for k, v in feed_dict.items() if 'name' not in k and 'ids' not in k}
        lidar_input_1 = inputs['lidar']
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            outputs_1, features_1, pseudo_label_output_1 = self.model(lidar_input_1)
            loss_1 = self.criterion(outputs_1, targets_1) if outputs_1.requires_grad else None

        if outputs_1.requires_grad:
            total_loss = self._compute_loss(feed_dict, features_1, targets_1, loss_1)
            self._backpropagate(total_loss)
            return {'outputs': outputs_1, 'targets': targets_1}
        else:
            return self._evaluate_step(feed_dict, outputs_1)

    def _compute_loss(self, feed_dict: Dict[str, Any], features_1: torch.Tensor, targets_1: torch.Tensor, loss_1: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss for training.

        Args:
            feed_dict (Dict[str, Any]): Input data dictionary.
            features_1 (torch.Tensor): Features from the first input batch.
            targets_1 (torch.Tensor): Targets for the first input batch.
            loss_1 (torch.Tensor): Loss from the first input batch.

        Returns:
            torch.Tensor: Combined loss value.
        """
        inputs_2 = feed_dict['lidar_2'].cuda()
        targets_2 = feed_dict['targets_2'].F.long().cuda(non_blocking=True)
        outputs_2, features_2, pseudo_label_output_2 = self.model(inputs_2)

        loss_2 = self.criterion(outputs_2, targets_2)
        pseudo_label_loss_1 = self.criterion(pseudo_label_output_1, targets_1)
        pseudo_label_loss_2 = self.criterion(pseudo_label_output_2, targets_2)

        features_1_normalized = nn.functional.normalize(features_1, dim=1)
        features_2_normalized = nn.functional.normalize(features_2, dim=1)

        prototype_features_1 = self._compute_prototype_features(features_1_normalized, targets_1)
        self.model.momentum_update_key_encoder(prototype_features_1, init=(self.global_step == 1))

        features_1, features_2 = equalize_samples(features_1, features_2)
        mi_loss = compute_mutual_information(self.mine_net, features_1, features_2)

        total_loss = loss_1 + self.lambda_param * (loss_2 + pseudo_label_loss_1 + pseudo_label_loss_2 + mi_loss)
        self._log_losses(loss_1, loss_2, pseudo_label_loss_1, pseudo_label_loss_2, mi_loss, total_loss)

        return total_loss

    def _compute_prototype_features(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute prototype features based on class labels.

        Args:
            features (torch.Tensor): Input features.
            targets (torch.Tensor): Class labels.

        Returns:
            torch.Tensor: Prototype features.
        """
        prototypes = torch.zeros((configs.data.num_classes, features.shape[1]), device=features.device)
        for i in range(configs.data.num_classes):
            mask = (targets == i)
            if mask.sum() > 0:
                prototypes[i] = features[mask].mean(dim=0)
        return prototypes + 1e-8

    def _log_losses(self, *losses: List[torch.Tensor]) -> None:
        """
        Log the training losses.

        Args:
            *losses (List[torch.Tensor]): Loss values to log.
        """
        names = ['loss_1', 'loss_2', 'pseudo_label_loss_1', 'pseudo_label_loss_2', 'mi_loss', 'total_loss']
        for name, loss in zip(names, losses):
            self.summary.add_scalar(name, loss.item())

    def _backpropagate(self, loss: torch.Tensor) -> None:
        """
        Perform backpropagation and optimization.

        Args:
            loss (torch.Tensor): Loss value for backpropagation.
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    def _evaluate_step(self, feed_dict: Dict[str, Any], outputs: torch.Tensor) -> Dict[str, Any]:
        """
        Process evaluation step during training.

        Args:
            feed_dict (Dict[str, Any]): Input data dictionary.
            outputs (torch.Tensor): Model predictions.

        Returns:
            Dict[str, Any]: Mapped outputs and targets.
        """
        invs = feed_dict['inverse_map']
        all_labels = feed_dict['targets_mapped']
        output_list, target_list = [], []

        for idx in range(invs.C[:, -1].max() + 1):
            scene_pts = (feed_dict['lidar'].C[:, -1] == idx).cpu().numpy()
            inv_map = invs.F[invs.C[:, -1] == idx].cpu().numpy()
            label_mask = (all_labels.C[:, -1] == idx).cpu().numpy()
            outputs_mapped = outputs[scene_pts][inv_map].argmax(1)
            targets_mapped = all_labels.F[label_mask]
            output_list.append(outputs_mapped)
            target_list.append(targets_mapped)

        return {'outputs': torch.cat(output_list, 0), 'targets': torch.cat(target_list, 0)}

    def save_model(self, path: str) -> None:
        """
        Save the current model state to a file.

        Args:
            path (str): Path to save the model.
        """
        assert path.endswith('.pt'), "Checkpoint save path must end with .pt"
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state_dict, path)

    def _after_epoch(self) -> None:
        """
        Actions to perform after each epoch.
        """
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary for saving checkpoints.

        Returns:
            Dict[str, Any]: State dictionary containing model, scaler, optimizer, and scheduler states.
        """
        return {
            'model': self.model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary from a checkpoint.

        Args:
            state_dict (Dict[str, Any]): State dictionary to load.
        """
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a previous checkpoint if available.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        pass

    def train(self,
              dataflow: DataLoader,
              *,
              num_epochs: Optional[int] = None,
              callbacks: Optional[List[Callback]] = None) -> None:
        """
        Start the training process.

        Args:
            dataflow (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train.
            callbacks (Optional[List[Callback]]): List of callbacks to use during training.
        """
        self.dataflow = dataflow
        self.steps_per_epoch = len(self.dataflow)
        self.num_epochs = num_epochs

        if callbacks is None:
            callbacks = []
        self.callbacks = Callbacks(callbacks)
        self.summary = Summary()

        try:
            self.callbacks.set_trainer(self)
            self.summary.set_trainer(self)

            self.epoch_num = 0
            self.global_step = 0

            train_time = time.perf_counter()
            self.before_train()

            while self.epoch_num < self.num_epochs:
                self.epoch_num += 1
                self.local_step = 0

                logger.info(f'Epoch {self.epoch_num}/{self.num_epochs} started.')
                epoch_time = time.perf_counter()
                self.before_epoch()

                for feed_dict in self.dataflow:
                    self.local_step += 1
                    self.global_step += 1

                    self.before_step(feed_dict)
                    output_dict = self.run_step(feed_dict)
                    self.after_step(output_dict)

                    self.trigger_step()

                self.after_epoch()
                logger.info(f'Training finished in {humanize.naturaldelta(time.perf_counter() - epoch_time)}.')

                self.trigger_epoch()
                logger.info(f'Epoch finished in {humanize.naturaldelta(time.perf_counter() - epoch_time)}.')

            logger.success(f'{self.num_epochs} epochs of training finished in {humanize.naturaldelta(time.perf_counter() - train_time)}.')
        except StopTraining as e:
            logger.info(f'Training was stopped by {str(e)}.')
        finally:
            self.after_train()


def get_pseudo_targets(output):
    """
    Get pseudo label targets from model output.

    Args:
        output (torch.Tensor): Model output.

    Returns:
        torch.Tensor: Pseudo label targets.
    """
    _, pseudo_targets = torch.max(output, dim=1)
    return pseudo_targets

def compute_pseudo_label_loss(pred, pseudo_targets):
    """
    Compute loss for pseudo label predictions.

    Args:
        pred (torch.Tensor): Predicted pseudo labels.
        pseudo_targets (torch.Tensor): Ground truth pseudo labels.

    Returns:
        torch.Tensor: Computed loss.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(pred, pseudo_targets)

def evaluate(val_loader, model):
    """
    Evaluate model on validation data.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to evaluate.
    """
    mIoU = MeanIoU(name='iou/test_', num_classes=19, ignore_label=255)
    mIoU.before_epoch()

    with torch.no_grad():
        for feed_dict in tqdm.tqdm(val_loader, ncols=0):
            inputs = {k: v.cuda() for k, v in feed_dict.items() if 'name' not in k}
            lidar_input = inputs['lidar']
            outputs = model(lidar_input)

            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            output_list, target_list = [], []

            for idx in range(invs.C[:, -1].max() + 1):
                scene_pts = (lidar_input.C[:, -1] == idx).cpu().numpy()
                inv_map = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                label_mask = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = outputs[scene_pts][inv_map].argmax(1)
                targets_mapped = all_labels.F[label_mask]
                output_list.append(outputs_mapped)
                target_list.append(targets_mapped)

            outputs = torch.cat(output_list, 0)
            targets = torch.cat(target_list, 0)
            assert not outputs.requires_grad, "produced grad, wrong"
            output_dict = {'outputs': outputs, 'targets': targets}
            mIoU.after_step(output_dict)
    mIoU.after_epoch()

class MINE(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) for computing mutual information.
    """

    def __init__(self, input_size=32):
        """
        Initialize the MINE module.

        Args:
            input_size (int): Size of the input features.
        """
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x, y):
        """
        Forward pass for computing mutual information.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Output after computing mutual information.
        """
        z = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(z))
        output = self.fc2(h)
        return output

def compute_mutual_information(mine_net, feat_1, feat_2):
    """
    Compute mutual information loss using MINE.

    Args:
        mine_net (MINE): MINE network instance.
        feat_1 (torch.Tensor): Feature set 1.
        feat_2 (torch.Tensor): Feature set 2.

    Returns:
        torch.Tensor: Mutual information loss.
    """
    T_joint = mine_net(feat_1, feat_2)
    feat_2_shuffled = feat_2[torch.randperm(feat_2.size(0), device=feat_2.device)]
    T_marginal = mine_net(feat_1, feat_2_shuffled)
    mi_loss = (T_joint.mean() - torch.log(torch.exp(T_marginal).mean() + 1e-8))
    return -mi_loss

def equalize_samples(feat_1, feat_2):
    """
    Equalize the number of samples between two feature sets.

    Args:
        feat_1 (torch.Tensor): Feature set 1.
        feat_2 (torch.Tensor): Feature set 2.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Equalized feature sets.
    """
    len_1, len_2 = feat_1.shape[0], feat_2.shape[0]
    if len_1 > len_2:
        return feat_1[:len_2], feat_2
    elif len_2 > len_1:
        return feat_1, feat_2[:len_1]
    return feat_1, feat_2
