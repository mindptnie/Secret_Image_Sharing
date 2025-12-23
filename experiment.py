
import os
import math
import torch
import time
from tqdm import tqdm
from model.base import BaseVAE
from torch import optim
from torch import tensor as Tensor
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import utility as util

import graph

class Experiment():
    def __init__(self, 
                vae: BaseVAE, 
                params: dict):
        self.vae = vae
        self.params = params
        self.train_loss = None
        self.learning_rate = None
        
    def training_step(self, batch):
        real_img = batch

        results = self.vae.forward(real_img)
        train_loss = self.vae.loss_function(*results,
                                            M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                            )

        return train_loss['loss']
    
    def validation_step(self, batch):
        real_img = batch

        results = self.vae.forward(real_img)
        val_loss = self.vae.loss_function(*results,
                                          M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                          )

        return val_loss['loss']
    
    def test_step(self, batch):
        real_img = batch

        results = self.vae.forward(real_img)
        test_loss = self.vae.loss_function(*results,
                                           M_N = 1, #al_img.shape[0]/ self.num_train_imgs,
                                           )

        return test_loss['loss']

    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.vae.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                            step_size = self.params['step_size'],
                                            gamma = self.params['scheduler_gamma'])
        
        return optimizer, scheduler
        
    def train(self, train_dataloader: DataLoader, optimizer, scheduler, device, val_dataloader: DataLoader=None, saved=[1,10], saved_path=None):
        
        loss_history = []
        val_history = []
        lr_history = []
        
        self.vae.to(device)
        
        scaler = GradScaler()
        
        print(f"Starting training for {self.params['max_epochs']} epochs on {device}...")
        total_start_time = time.time()

        for epoch in range(self.params['max_epochs']):
            self.vae.train()
            
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
                
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)

                scaler.update()
                                
                epoch_loss += loss.item()
                
                batch_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
                
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            loss_history.append(avg_epoch_loss)
            
            #=========================== Validation Step ===========================#
            if val_dataloader is not None:
                self.vae.eval()
                val_epoch_loss = 0
                val_pbar = tqdm(enumerate(val_dataloader),
                                desc=f"Epoch {epoch+1} [Validate]",
                                leave=False,
                                total=len(val_dataloader))
                
                with torch.no_grad():
                    for batch in val_pbar:
                        batch = batch.to(device)
                        with autocast(device_type=self.params['device_type']):
                            val_loss = self.validation_step(batch) 
                        val_epoch_loss += val_loss.item()
                        
                        
                        val_pbar.set_postfix({'val_loss': f'{val_loss.item():.4f}'})

                avg_val_loss = val_epoch_loss / len(val_dataloader)
                val_history.append(avg_val_loss)
            
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            log_msg = (f"Epoch [{epoch+1}/{self.params['max_epochs']}] | "
                       f"Avg. Loss: {avg_epoch_loss:.8f} | ")
            if val_dataloader is not None:
                log_msg += f"Avg. Val Loss: {avg_val_loss:.8f} | "
            log_msg += (f"LR: {current_lr:.8f} | ")
            tqdm.write(log_msg)

            # save model
            if epoch+1 in saved and saved_path:
                model_save_path = util.join_paths(saved_path+"_model"+"_epoch"+str(epoch+1)+".pth")
                print(f"Model saved to: {model_save_path}")
                torch.save(self.vae, model_save_path)

        total_end_time = time.time()
        total_duration_sec = total_end_time - total_start_time
        
        print("Training finished!")
        print(f"Total training time: {total_duration_sec / 60:.2f} minutes ({total_duration_sec:.2f} seconds)")
        
        return {
            "train_loss": loss_history,
            "val_loss": val_history if val_dataloader is not None else None, 
            "learning_rate": lr_history
        }
        
        