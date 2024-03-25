from .simclr_utils import save_config_file, accuracy, save_checkpoint

import logging
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

class SimCLR(object):
    def __init__(self, model, model_name, scheduler, optimizer, device, epochs=200, batch_size=256, lr=3e-4, weight_decay=1e-4, fp16_precision=True, log_n_steps=100, temp=0.07):
        """
        SimCLR

        Args:
            model (torch model): torch model
            model_name (str): model name, VGG16 or ResNet
            scheduler (torch scheduler): torch scheuduler
            optimizer (torch optimizer): torch optimizer
            device (str): 'cpu' / 'cuda' / 'mps'
            epochs (int, optional): number of epoch for simclr. Defaults to 200.
            batch_size (int, optional): batch size for simclr. Defaults to 256.
            lr (float, optional): initial learning rate for Adam optimizer. Defaults to 3e-4.
            weight_decay (float, optional): weight decay for Adam optimzer. Defaults to 1e-4.
            fp16_precision (bool, optional): if True, use 16-bit precision GPU training. Defaults to True.
            log_n_steps (int, optional): Log every n steps. Defaults to 100.
            temp (float, optional): softmax temperature. Defaults to 0.07.
        """
        self.model = model
        self.model_name = model_name
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.fp16_precision = fp16_precision
        self.log_n_steps = log_n_steps
        self.temp = temp
        
        self.metadata = (f"model_name: {self.model_name}, "\
                         f"epochs: {self.epochs}, "\
                         f"batch_size: {self.batch_size}, "\
                         f"lr: {self.lr}, "\
                         f"weight_decay: {self.weight_decay}, "\
                         f"fp16_precision: {self.fp16_precision}, "\
                         f"log_n_steps: {self.log_n_steps}, "\
                         f"temperature: {self.temp}, "\
                         f"optimizer: {type(self.optimizer).__name__}, "\
                         f"scheduler: {type(self.scheduler).__name__}, "\
                         f"device: {self.device}")

        self.n_views = 2
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'simclr_training.log'), level=logging.DEBUG)

    def info_nce_loss(self, features):
        # labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = torch.cat([torch.arange(int(features.size(0)/2)) for i in range(self.n_views)], dim=0)
        
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temp
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.metadata)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.epochs} epochs.")
        logging.info(f"Training with: {self.device} device.")
        logging.info(f"SimCLR Setting: {self.metadata}\n")

        for epoch_counter in range(self.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                with autocast(enabled=self.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.log_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.epochs)
        save_checkpoint({
            'epoch': self.epochs,
            'model_name': self.model_name,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
