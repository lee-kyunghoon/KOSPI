import os
import numpy as np

'''
from app.utils import init_model, init_optim, load_checkpoint
from src.data.data_manager import init_data
from src.utils.log import (
    AverageMeter,
    get_logger,
    CSVLogger,
    grad_logger,
    adamw_logger
    )
'''

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from tqdm import tqdm

#logger = get_logger(__name__)


from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CSVWindowDataset(Dataset):
    """
    Dataset for windowed time-series from CSV files.

    - Uses rows 2..N of the original CSV as data (row 1 = header).
    - Uses columns 2..10 (1-based) => df.iloc[:, 1:10] (0-based).
    - For given input_seq_len, output_seq_len, each item i corresponds to:
        start row = num_i = 2 + i   (in your original description)
        x: [start .. start+input_seq_len-1], cols 2..10
        y: [start+input_seq_len .. start+input_seq_len+output_seq_len-1], cols 2..10
    """
    def __init__(self, csv_path: str, input_seq_len: int, output_seq_len: int):
        super().__init__()
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        # Read CSV: row 1 is header; data rows start at original row 2.
        df = pd.read_csv(csv_path)

        # Columns 2~10 (1-based) => 1:10 (0-based, end-exclusive)
        values = df.iloc[:, 1:10].to_numpy(dtype="float32")  # shape [T, 9]

        self.data = torch.from_numpy(values)                 # [T, 9]
        T = self.data.shape[0]

        # Number of valid starting positions (num_i) in your description:
        # num_i ranges from 2 to  (N - (input_seq_len + output_seq_len - 1))
        # => dataset index i = num_i - 2 ranges 0 .. T - (input+output)
        self.num_windows = T - (input_seq_len + output_seq_len) + 1
        if self.num_windows <= 0:
            raise ValueError(
                f"Not enough rows ({T}) for input_seq_len={input_seq_len} and "
                f"output_seq_len={output_seq_len}"
            )

    def __len__(self):
        # This equals (2960 - 1) - (input+output) + 1 for your train.csv
        return self.num_windows

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of range [0, {self.num_windows-1}]")

        # This dataset index `idx` corresponds to your num_i = 2 + idx
        start = idx
        in_end = start + self.input_seq_len
        out_end = in_end + self.output_seq_len

        x = self.data[start:in_end, :]   # [input_seq_len, 9]
        y = self.data[in_end:out_end, :] # [output_seq_len, 9]

        return x, y


def make_dataloader(csv_path: str,
                    batch: int,
                    input_seq_len: int,
                    output_seq_len: int,
                    shuffle: bool) -> DataLoader:
    """
    Returns a PyTorch DataLoader that yields:
        x: [batch, input_seq_len,  9]
        y: [batch, output_seq_len, 9]

    - shuffle=False: indices go sequentially: num_i = 2, 3, 4, ...
    - shuffle=True:  each full pass (epoch) is a random permutation of all
                     valid num_i, without repetition (exactly what you asked).
    """
    dataset = CSVWindowDataset(csv_path, input_seq_len, output_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,  # no repetition within one pass (epoch)
        drop_last=False,  # keep last partial batch if any
    )
    return loader

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)
        
class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # target, pred: same shape
        denom = torch.clamp(target.abs(), min=self.eps)
        return torch.mean(torch.abs((target - pred) / denom))

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
import math


def build_optimizer_and_scheduler(model: nn.Module, 
                                  initial_lr: float,      # peak LR (after warmup)
                                  final_lr: float,        # LR at the final epoch
                                  weight_decay: float,
                                  num_epochs: int,
                                  grad_clip: float | None,    # max norm for grad clipping (None = no clipping)
                                  use_warmup: bool,
                                  ):
    """
    Returns:
        optimizer: Adam optimizer with weight decay
        scheduler: LambdaLR doing (optional) warmup + cpsome decay
    """
    optimizer = Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
    )

    # we’ll use 10% of epochs for warmup if enabled
    warmup_epochs = int(0.1 * num_epochs) if use_warmup else 0
    
    # final_lr = initial_lr * min_factor
    min_factor = final_lr / initial_lr

    def lr_lambda(epoch: int):
        # epoch is 0-based
        # Warm-up Phase
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # linear warmup from 0 -> initial_lr
            return float(epoch + 1) / float(warmup_epochs)

        # Cosine decay phase after warm-up
        total_cosine_epochs = max(1, num_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / total_cosine_epochs
        progress = min(max(progress, 0.0), 1.0)

        # cosine goes from 1 -> 0 smoothly
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # scale between 1.0 (start) and min_factor (end)
        factor = min_factor + (1.0 - min_factor) * cosine
        return factor

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
    
import logging

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger 

def main(args):

    train_batch_size = 8 # number of data samples in one data batch. 
    input_seq_len = 30 # use 30 for feeding past 30 day data
    output_seq_len = 5 # use 5 for making prediction of future 5 day data
    
    
    train_csv_path = "./data/train.csv" # example. update later to indicate right path
    valid_csv_path = "./data/valid.csv" # example. update later to indicate right path
    test_csv_path = "./data/test.csv"   # example. update later to indicate right path
    
    train_loader = make_dataloader(train_csv_path, train_batch_size, input_seq_len, output_seq_len, shuffle=True)
    valid_loader = make_dataloader(valid_csv_path, valid_batch_size, input_seq_len, output_seq_len, shuffle=False)
    test_loader  = make_dataloader(test_csv_path,  test_batch_size, input_seq_len, output_seq_len, shuffle=False)
    # On the format of outputs of data loaders 
    # output is of the form (x,y). fetch (x,y) as x, y = next(iter(train_loader)), or use loop like : for x, y in train_loader:
    # x have shape [batch, input_seq_len, 9] where x[i,j,1] ~ x[i,j,9] contains 
    # KOSPI_Close	KOSPI_Open	KOSPI_High	KOSPI_Low	KOSPI_Volume	NASDAQ_Close	US10Y_Yield	VIX_Close	USD_KRW
    # of j-th (past) date in the input data sequence, of i-th data sample in the batch. (each data sample is input_seq_len - long data sequence of 9 dim vector)

    # y have shape [batch, input_seq_len, 9] where y[i,j,1] ~ y[i,j,9] contains 
    # KOSPI_Close	KOSPI_Open	KOSPI_High	KOSPI_Low	KOSPI_Volume	NASDAQ_Close	US10Y_Yield	VIX_Close	USD_KRW
    # of j-th (future) date in the input data sequence, of i-th data sample in the batch. (each data sample is input_seq_len - long data sequence of 9 dim vector)

    ## MODEL
    import .cnntrans # importing implemented CNN+transformer structure. update later to import CNN+transformer structure from right path 
    import .ae       # importing implemented CNN+transformer structure. update later to import CNN+Autoencoder structure from right path 


    # models. pick one between CNNTrans or CNNAutoencoder()   
    model = CNNTrans()
    # mdoel = CNNAutoencoder() # arbitrary class name written at 2025-11-13. update later   

    ## OPTIMAZATION
    from torch import nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR
    from dataclasses import dataclass
    
    
    
    # optimizer and schedulers initialized here.  
    
    initial_lr=1e-2
    final_lr=1e-4
    weight_decay=1e-4
    num_epochs=150
    grad_clip=1.0       
    use_warmup=True
    
    optimizer, scheduler = build_optimizer_and_scheduler(model, 
                                                        initial_lr=initial_lr,
                                                        final_lr=final_lr,
                                                        weight_decay=weight_decay,
                                                        num_epochs=num_epochs,
                                                        grad_clip=grad_clip, # clip gradients to max norm 1.0; set None to disable
                                                        use_warmup=use_warmup,
    )
    if type(model) is CNNAutoencoder:
        optimizer_AE, scheduler_AE = build_optimizer_and_scheduler(model.autoencoder, # note that it is assumed here that autoencoder object is defined inside the model and accessible via model.autoencoder 
                                                                  initial_lr=initial_lr,
                                                                  final_lr=final_lr,
                                                                  weight_decay=weight_decay,
                                                                  num_epochs=num_epochs,
                                                                  grad_clip=grad_clip, # clip gradients to max norm 1.0; set None to disable
                                                                  use_warmup=use_warmup,
        )
    else : 
        optimizer_AE = None
        scheduler_AE = None

    ## LOSS, pick one for the below training loop
    mae_loss = nn.L1Loss() # MAE loss
    rmse_loss = RMSELoss() # RMSE loss
    mape_loss = MAPELoss() # MAPE loss
    # loss_fn = mae_loss
    # loss_fn = rmse_loss
    # loss_fn = mape_loss

    ## LOG
    # set save_path <- directory where the log file is saved
    # example save_path is declared below. 
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y_%m_%d_%H_%M_%S")
    save_path = './experiment_logs/'+ stamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = get_logger(logpath=os.path.join(save_path, '/logs'), filepath=os.path.abspath(__file__))
    logger.propagate = False


    ## META
    seed = 2025 # arbitrary example 
    save_period = 5 # arbitrary example

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    ## SET SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.benchmark = True

    ## SET DEVICE
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
 
    ## train loop
    model.train()
    loss_meter = AverageMeter()
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
    
            optimizer.zero_grad()
            pred = model(x)      
            
            if grad_clip  is not None : 
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            loss = loss_fn(pred, y[:,:,0]) # Here it is assumed that output of the model have shape of [batch_size, 5]
            loss.backward()
            
            # we are training Autoencoder seperately so zero-ing out the grad of autoencoder 
            if isinstance(model, CNNAutoencoder): 
                optimizer_AE.zero_grad()
            optimizer.step()
            optimizer.zero_grad()
            
            #learning of autoencoder is done here
            # note that we are assuming model.autoencoder and model.encoder is accessible here, for CNN+autoencoder structure
            if isinstance(model, CNNAutoencoder): 
                optimizer_AE.zero_grad()
                Autoencoder_input = model.encoder(x)
                Autoencoder_output = model.autoencoder(Autoencoder_input)
                AE_loss = F.mse_loss(Autoencoder_output, Autoencoder_input)
                AE_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), grad_clip)
                optimizer_AE.step()
                optimizer_AE.zero_grad()

        scheduler.step()
        if isinstance(model, CNNAutoencoder): 
            scheduler_AE.step()

        ## logging and checkpoint saving
        loss_meter.update(loss.item(), batch_size)
                        
        logger.info('avg. loss %.3f' % loss_meter.avg)

        if (epoch % 5 == 0) or (epoch == num_epochs -1 ): # arbitrarily set period for checkpoint saving to 5. update later if needed
            checkpoint_path = os.path.join(save_path, '/checkpoint_epoch_'+str(epoch)+'.pth' )
            
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "optimizer_AE_state": optimizer_AE.state_dict() if optimizer_AE is not None else None,
                "scheduler_AE_state": scheduler_AE.state_dict() if scheduler_AE is not None else None,
                "initial_lr": initial_lr,
                "final_lr"  : final_lr,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs ,
                "grad_clip" : grad_clip,       
                "use_warmup": use_warmup
            }
            
            torch.save(checkpoint, checkpoint_path )
