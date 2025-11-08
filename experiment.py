
import os
import math
import torch
import time
from tqdm import tqdm
from vae import VariationalAutoencoder
from torch import optim
from torch import tensor as Tensor
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import graph

class Experiment():
    def __init__(self, 
                vae: VariationalAutoencoder, 
                params: dict):
        self.vae = vae
        self.params = params
        
    def training_step(self, batch):
        real_img = batch

        results = self.vae.forward(real_img)
        train_loss = self.vae.loss_function(*results,
                                            M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                            )

        return train_loss['loss']

    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.vae.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                            step_size = self.params['step_size'],
                                            gamma = self.params['scheduler_gamma'])
        
        return optimizer, scheduler
    
    def train(self, train_dataloader: DataLoader, optimizer, scheduler, device):
        
        loss_history = []
        lr_history = []
        
        self.vae.to(device)
        self.vae.train()
        
        scaler = GradScaler()
        
        print(f"Starting training for {self.params['max_epochs']} epochs on {device}...")
        total_start_time = time.time()

        for epoch in range(self.params['max_epochs']):
            epoch_loss = 0
            batch_pbar = tqdm(train_dataloader, 
                              desc=f"Epoch {epoch+1}/{self.params['max_epochs']}", 
                              leave=False)
            for batch in batch_pbar:
                
                batch = batch.to(device)
                
                optimizer.zero_grad()
                
                with autocast(device_type=self.params['device_type']):
                    loss = self.training_step(batch) 
                
                scaler.scale(loss).backward()
                
                scaler.step(optimizer)

                scaler.update()
                                
                epoch_loss += loss.item()
                
                batch_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
                
            scheduler.step()
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            
            loss_history.append(avg_epoch_loss)
            lr_history.append(current_lr)
            
            tqdm.write(f"Epoch [{epoch+1}/{self.params['max_epochs']}], Avg. Loss: {avg_epoch_loss:.8f}, LR: {current_lr:.8f}")
        
        
        total_end_time = time.time()
        total_duration_sec = total_end_time - total_start_time
        
        print("\nTraining finished!")
        print(f"Total training time: {total_duration_sec / 60:.2f} minutes ({total_duration_sec:.2f} seconds)")
        
        graph.loss_curve(epochs=self.params['max_epochs'],loss_history=loss_history)
        graph.learning_rate(epochs=self.params['max_epochs'],lr_history=lr_history)
        