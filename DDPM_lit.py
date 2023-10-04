import yaml
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from lib.model import ConditionalModel
from lib.diffusion import GaussianDiffusion, make_beta_schedule
from lib.data_helpers import PhotonsDataset



class DDPM_lit(pl.LightningModule):
    def __init__(self, config):
        super(DDPM_lit, self).__init__()
        self.config = config
        self.model = ConditionalModel(n_steps=self.config["n_steps"])
        self.lr = self.config["learning_rate"]
        self.betas = make_beta_schedule(schedule=self.config["beta_scheduler_params"]["beta_schedule"], 
                                        n_timesteps=self.config["n_steps"], 
                                        start=self.config["beta_scheduler_params"]["start"], 
                                        end=self.config["beta_scheduler_params"]["end"])
        self.diffusion = GaussianDiffusion(betas=self.betas)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        return self.diffusion.p_sample_loop(self.model, x.shape)
    
    def training_step(self, batch, batch_idx):
        loss = self.diffusion.noise_estimation_loss(self.model, batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_loss = self.diffusion.noise_estimation_loss(self.model, batch)
        self.log("val_loss", val_loss)


if __name__ == "__main__":
    
    with open("config/ddpm_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    model = DDPM_lit(config)

    checkpoint_callback = ModelCheckpoint(**config['checkpoint_params'])
    learning_rate_monitor_callback = LearningRateMonitor()
    csv_logger = pl_loggers.CSVLogger(**config['logger'])
    
    trainer = pl.Trainer(fast_dev_run=False,
                         accelerator='gpu',
                         devices=config["n_gpus"],
                         max_epochs=config["n_epochs"],
                         gradient_clip_val=1.,
                         callbacks=[learning_rate_monitor_callback, checkpoint_callback],
                         logger=csv_logger)
    
    photons_data = config["dataset_params"]["data_path"] + config["dataset_params"]["filename"]
    photon_data = PhotonsDataset(photons_data, batch_size=config["dataset_params"]["batch_size"], 
                                 shuffle_train=False, num_workers=2, validation_fraction=0.3, )
    train_loader = photon_data.train_dataloader()
    val_loader = photon_data.val_dataloader()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


