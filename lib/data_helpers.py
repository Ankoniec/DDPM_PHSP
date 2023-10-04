import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def plot_photons_histograms(dataset:np.ndarray, axs, figs, log_ax:bool=False, 
                            label='original', title:str=None, data_color:str='crimson', 
                            a:float=0.8, ylabel:str='Density', hist_type:str='bar', lw:float=0.5
                            ) -> None:
    assert dataset.shape[1]==6

    columns = ['E [MeV]','X [cm]', 'Y [cm]', 'Px [MeV]', 'Py [MeV]', 'Pz [MeV]']

    for i, var in enumerate(columns):
        axs.flatten()[i].hist(dataset[:, i], bins=300, color=data_color, alpha=a, label=label, density=True, histtype=hist_type, linewidth=lw)
        axs.flatten()[i].xaxis.set_label_coords(.5, -.1)
        axs.flatten()[i].set_xlabel(var, fontsize=12)

    axs[0,2].legend(loc='lower center')
    axs.flatten()[0].set_ylabel(ylabel, fontsize=12)
    axs.flatten()[3].set_ylabel(ylabel, fontsize=12)

    if log_ax:
        axs[0,0].set_yscale('log')
        axs[0,1].set_yscale('log')
        axs[0,2].set_yscale('log')
        axs[1,0].set_yscale('log')
        axs[1,1].set_yscale('log')
        axs[1,2].set_yscale('log')

    if title:
        figs.suptitle(title, fontsize=16)



class PhotonsDataset(Dataset):
    def __init__(self, data_path, batch_size, num_workers, validation_fraction, shuffle_train, random_seed=3838, transform=None):
        self.data = self.setup_data(data_path)
        self.n_samples = self.data.shape[0]
        self.transform = transform
        self.columns = ['E [MeV]', 'X [cm]', 'Y [cm]', 'Px [MeV]', 'Py [MeV]', 'Pz [MeV]']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.random_seed = random_seed
        self.X_train, self.X_validation = train_test_split(
            self.data, test_size=validation_fraction, random_state=random_seed, shuffle=True)

    def __getitem__(self, index):
        sample = self.data[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

    def setup_data(self, data_path):
        photons = np.load(data_path)
        mmsc = MinMaxScaler()
        photons_mm = mmsc.fit_transform(photons)
        return photons_mm
    
    def train_dataloader(self):
        return DataLoader(dataset=self.X_train, 
                          batch_size=self.batch_size, 
                          shuffle=self.shuffle_train, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.X_validation, 
                          batch_size=self.batch_size, 
                          shuffle=self.shuffle_train, 
                          num_workers=self.num_workers, 
                          pin_memory=True)