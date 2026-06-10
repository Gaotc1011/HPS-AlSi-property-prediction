import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision
import re
from readvtk import readvtk
import matplotlib.pyplot as plt

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class TinyData(Dataset):
    def __init__(self, src="../data/labeled_vtks", seed=40, *args, **kwargs):
        self.setname = kwargs["setname"] if "setname" in kwargs else "train"
        self.seed = seed
        self.dataname = kwargs["dataname"] if "dataname" in kwargs else "vtk_and_performance"
        self.performance_columns = kwargs["property_columns"] if "property_columns" in kwargs else []

        self.overall_dataset_dir = src

        # 分别读取 phas 和 comp
        self.phas_vtkfiles = sorted([f for f in os.listdir(self.overall_dataset_dir) if f.endswith(".vtk") and "phas" in f.lower()])
        self.comp_vtkfiles = sorted([f for f in os.listdir(self.overall_dataset_dir) if f.endswith(".vtk") and "comp" in f.lower()])

        # 对应的性能 CSV
        self.csvfiles = [f for f in os.listdir(self.overall_dataset_dir) if f.endswith('.csv')]
        csv_path = os.path.join(self.overall_dataset_dir, self.csvfiles[0])
        self.csv_data = pd.read_csv(csv_path)

        # 保证索引对应
        assert len(self.phas_vtkfiles) == len(self.comp_vtkfiles), "phas 和 comp 文件数量不一致"

        self.indices = np.arange(len(self.phas_vtkfiles))
        np.random.seed(self.seed)
        np.random.shuffle(self.indices)

        self.train_indices = self.indices[:int(len(self.indices) * 0.8)]
        self.test_indices  = self.indices[int(len(self.indices) * 0.8):int(len(self.indices) * 0.9)]
        self.val_indices   = self.indices[int(len(self.indices) * 0.9):]

    def __len__(self):
        if self.setname == 'train':
            return len(self.train_indices)
        elif self.setname == 'test':
            return len(self.test_indices)
        elif self.setname == 'val':
            return len(self.val_indices)
        return len(self.phas_vtkfiles)

    def __getitem__(self, idx):
        indices_to_use = getattr(self, f"{self.setname}_indices", self.indices)
        real_idx = indices_to_use[idx]

        phas_file  = self.phas_vtkfiles[real_idx]
        comp_file = self.comp_vtkfiles[real_idx]

        # 读取 phas
        result_p, dims_p = readvtk(os.path.join(self.overall_dataset_dir, phas_file))
        data_p = np.array(result_p["scalars"]).reshape((dims_p[2], dims_p[1], dims_p[0]))

        # 读取 comp
        result_e, dims_e = readvtk(os.path.join(self.overall_dataset_dir, comp_file))
        data_e = np.array(result_e["scalars"]).reshape((dims_e[2], dims_e[1], dims_e[0]))

        # 转 tensor，并保持 (H, W)，不要额外 permute
        toTensor = torchvision.transforms.ToTensor()
        torch_p = toTensor(data_p.astype(np.float32)).unsqueeze(0)  # shape: (1, H, W)
        torch_e = toTensor(data_e.astype(np.float32)).unsqueeze(0)  # shape: (1, H, W)

        # 拼接成 2 通道输入
        torch_data = torch.cat([torch_p, torch_e], dim=0)  # shape: (2, H, W)
        
        # 找到对应的性能指标
        if self.dataname == "vtk_and_performance":
            id = int(re.findall(r'\d+', phas_file)[0])  # 用 phas 文件名提取 id
            perf_values = {
                col: float(self.csv_data[self.csv_data['id'] == id][col].iloc[0]) 
                for col in self.performance_columns
            }
            perf_tensors = {col: torch.tensor(val, dtype=torch.float32) for col, val in perf_values.items()}
            return torch_data.squeeze(), perf_tensors, (phas_file, comp_file)
        else:
            return torch_data.squeeze(), (phas_file, comp_file)


    
def visualize_torch_data(torch_data, title="Visualization of Torch Data"):
    """
    Visualize the torch_data tensor using matplotlib.
    Args:
        torch_data (torch.Tensor): The tensor to visualize.
        title (str): Title of the plot.
    """
    # Convert torch tensor to numpy array
    np_data = torch_data.squeeze().detach().cpu().numpy()
   # Check the dimensionality of the data
    if np_data.ndim == 2:  # 2D data (single channel)
        plt.imshow(np_data, cmap='gray')
    elif np_data.ndim == 3:  # 3D data (volume)
        # Plot middle slice along the first axis
        mid_slice = np_data.shape[0] // 2
        plt.imshow(np_data[mid_slice], cmap='gray')
    else:
        raise ValueError("Unsupported data dimensionality for visualization.")

    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
        # create dataset
    train_dataset = TinyData(src="./data_try-4", setname = "all", dataname="vtk_and_performance", property_columns=["HV"])
    # test_dataset = TinyData(src="../data/labeled_vtks")

    # build dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=110, shuffle=True)
    for data in train_dataloader:
        torch_data,property,selected_filename= data
        print(torch_data.shape)
        # print(selected_filename)
        # print(property)
        visualize_torch_data(torch_data[1], title=f"Visualization of {selected_filename[1]}")
        exit(0)