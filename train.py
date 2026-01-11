import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import argparse

from data import create_dataloaders
from model import ProteinBERT
from config import Config

class LearningRateScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * (self.warmup_steps ** (-1.5))
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * Config.LEARNING_RATE
        return lr

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'vocab_size': Config.VOCAB_SIZE,
            'd_model': Config.D_MODEL,
            'n_layers': Config.N_LAYERS,
            'n_heads': Config.N_HEADS,
            'd_ff': Config.D_FF,
            'max_length': Config.MAX_LENGTH,
            'dropout': Config.DROPOUT
        }
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            total_samples += input_ids.size(0)
            
            predictions = torch.argmax(logits, dim=-1)
            mask = (labels != -100)
            correct_predictions += ((predictions == labels) & mask).sum().item()
            total_predictions += mask.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return avg_loss, accuracy

def train_model(resume_from=None):
    device = Config.DEVICE
    print(f"device: {device}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        max_length=Config.MAX_LENGTH,
        max_files=Config.MAX_FILES
    )
    
    print(f"train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    print("="*25)

    model = ProteinBERT(
        vocab_size=Config.VOCAB_SIZE,
        d_model=Config.D_MODEL,
        n_layers=Config.N_LAYERS,
        n_heads=Config.N_HEADS,
        d_ff=Config.D_FF,
        max_length=Config.MAX_LENGTH,
        dropout=Config.DROPOUT
    )
    torch.compile(model)
    model = model.to(device)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = LearningRateScheduler(optimizer, Config.D_MODEL, Config.WARMUP_STEPS)
    
    # resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if resume_from and os.path.exists(resume_from):
        start_epoch, global_step, _ = load_checkpoint(resume_from, model, optimizer)
        print(f"resume from epoch {start_epoch}, step {global_step}")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True) # dir for checkpoints
    
    # TensorBoard stuff
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/protein_bert_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    model.train()
    for epoch in range(start_epoch, Config.MAX_EPOCHS):
        epoch_loss = 0
        epoch_samples = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            
            optimizer.step()
            lr = scheduler.step()
            
            epoch_loss += loss.item()
            epoch_samples += input_ids.size(0)
            global_step += 1
            
            # log 
            if global_step % Config.LOG_INTERVAL == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"poch {epoch+1}/{Config.MAX_EPOCHS}, "
                      f"Step {global_step}, "
                      f"batch {batch_idx+1}/{len(train_loader)}, "
                      f"loss: {loss.item():.4f}, "
                      f"avg loss: {avg_loss:.4f}, "
                      f"LR: {lr:.2e}")
                
                writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
                writer.add_scalar('Learning_Rate', lr, global_step)
            
            # checkpoint
            if global_step % Config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(Config.MODEL_DIR, f"checkpoint_step_{global_step}.pt")
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss.item(), checkpoint_path)
        
        # end of epoch evaluation
        epoch_time = time.time() - start_time
        avg_train_loss = epoch_loss / len(train_loader)
        
        print(f"\nepoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"average training loss: {avg_train_loss:.4f}")
        
        # validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, device)
        
        print(f"val loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        
        # tensorboard
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/Val_Epoch', val_accuracy, epoch)
        
        # save epoch checkpoint
        checkpoint_path = os.path.join(Config.MODEL_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, avg_train_loss, checkpoint_path)
        
        model.train()
        print("-" * 80)
    
    # save
    final_path = os.path.join(Config.MODEL_DIR, "final_model.pt")
    save_checkpoint(model, optimizer, scheduler, Config.MAX_EPOCHS, global_step, avg_train_loss, final_path)
    
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Protein BERT')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--max_files', type=int, default=Config.MAX_FILES, help='Maximum number of files to process')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=Config.MAX_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='Learning rate')

    args = parser.parse_args()

    # Update config with command line arguments
    Config.MAX_FILES = args.max_files
    Config.BATCH_SIZE = args.batch_size
    Config.MAX_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr

    
    train_model(resume_from=args.resume)
