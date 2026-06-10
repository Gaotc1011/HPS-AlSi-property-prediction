
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from dataloader_2 import TinyData
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a convolution neural network
class ModelCNN(nn.Module):
    def __init__(self, output_dim):
        super(ModelCNN, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=4, stride=2, padding=1), # 600*600*2-->300*300*8
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1), # 300*300*8-->150*150*16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1), # 150*150*16 -->75*75*16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),# 75*75*16 -->37*37*8
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=8, out_channels=2, kernel_size=4, stride=2, padding=1),# 37*37*8 -->18*18*2
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(18*18*2, 256)
        )

        self.predictor = nn.Sequential(
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(64,output_dim),
        )

    def forward(self, input):
        latent = self.extractor(input)
        output = self.predictor(latent)

        return output,latent
    
    def save(self, prefix_path):
        # extractor
        extractor_path = os.path.join(prefix_path, "extractor.pth")
        torch.save(self.extractor.state_dict(), extractor_path)
        # predictor
        predictor_path = os.path.join(prefix_path, "predictor.pth")
        torch.save(self.predictor.state_dict(), predictor_path)


    def load_extractor(self, prefix_path):
        # extractor
        extractor_path = os.path.join(prefix_path, "extractor.pth")
        self.extractor.load_state_dict(torch.load(extractor_path))
        # 固定extractor参数，使其不参与后续的训练
        # for param in self.extractor.parameters():
        #     param.requires_grad = False

    def load_predictor(self, prefix_path):
        # predictor
        predictor_path = os.path.join(prefix_path, "predictor.pth")
        self.predictor.load_state_dict(torch.load(predictor_path))
  


class Training_Autoencoder(object):
    def __init__(self, data_dir="./train-data", model_dir="./model-results/"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_loss = 100
        self.seed = 70
        self.property_columns = ["UTS","YS","HV"]
        self.model = ModelCNN(output_dim = len(self.property_columns))
        # 将模型移动到GPU上
        self.model.to(device)

    def load_dataset(self):
        # create dataset
        train_dataset = TinyData(src=self.data_dir, setname="all", seed = self.seed, property_columns = self.property_columns, dataname = "vtk_and_performance")

        # build dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
        
    def train(self, maxiter=100, lr=0.0001, restart=False):
        loss_func = torch.nn.L1Loss()
        # loss_func = torch.nn.CrossEntropyLoss()
        optimizer = self.load_optimizer(lr = lr, restart = restart)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        cnt_epochs = maxiter
        train_loss_history = []
        
        for cnt in range(cnt_epochs):
            total_loss_train = []

            for data in self.train_dataloader:
                phase, label_property, vtkfilename = data
                # 动态拼接性能指标
                label_property = torch.cat([label_property[prop].unsqueeze(-1) for prop in self.property_columns], dim=-1).to(device)

                phase = phase.to(device)
                # print(phase.shape)
                output, feature = self.model(phase)
                optimizer.zero_grad()
                loss = loss_func(output, label_property)
                loss.backward()
                optimizer.step()
                total_loss_train.append(loss.item())

            print("Epoch:", cnt, "Train Loss:", np.mean(total_loss_train))
            
            train_loss_history.append(np.mean(total_loss_train))


            if np.mean(total_loss_train) < self.best_loss:
                self.best_loss = np.mean(total_loss_train)
                self.save_model()
                print("save model %s"%cnt)
                self.save_optimizer(optimizer)                

            # print("Epoch: %d, loss: %.10f, best_loss: %.10f" % (cnt, np.mean(total_loss_train), self.best_loss))

        df_loss = pd.DataFrame({ 'train_loss': train_loss_history})
        df_loss.to_csv(os.path.join(self.model_dir, "loss_history-hardness.csv"), index=False)
  
        plt.figure()
        plt.plot(range(len(train_loss_history)), train_loss_history,label = "training")
        # plt.plot(range(len(test_loss_history)), test_loss_history,label = "testing")
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.model_dir, "loss_history.jpg"), dpi=500)
        plt.clf()
        plt.close()

    def save_model(self):
        # torch.save(self.model.state_dict(), os.path.join(self.model_dir, "ae.pth"))
        self.model.save(self.model_dir)

    def load_extractor(self):
        self.model.load_extractor(self.model_dir)
    
    def load_predictor(self):
        self.model.load_predictor(self.model_dir)
    
    def save_optimizer(self, optimizer):
        torch.save(optimizer.state_dict(), os.path.join(self.model_dir, "optimizer.pth"))

    def load_optimizer(self, lr=0.01, restart=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if restart:
            optimizer.load_state_dict(torch.load(os.path.join(self.model_dir, "optimizer.pth")))
        return optimizer

    def infer(self):   
        preds = {prop+"_true": [] for prop in self.property_columns}
        preds.update({prop+"_pred": [] for prop in self.property_columns})
        
        for data in self.train_dataloader:
            phase, label_property, vtkfilename = data
            # 拼接 label
            label_tensor = torch.cat(
                [label_property[prop].unsqueeze(-1) for prop in self.property_columns], dim=-1
            )
            phase = phase.to(device)
            output, feature = self.model(phase)
            output = output.detach().cpu().numpy()
            label_tensor = label_tensor.detach().cpu().numpy()

            # if label_tensor.ndim == 1:
            #     label_tensor = label_tensor[:, None]
            # if output.ndim == 1:
            #     output = output[:, None]

            # 遍历 batch 和属性
            for b in range(output.shape[0]):# batch 内循环
                for i, prop in enumerate(self.property_columns):
                    preds[prop+"_true"].append(label_tensor[b, i].item())
                    preds[prop+"_pred"].append(output[b, i].item())

        # 存成 DataFrame
        df_pred = pd.DataFrame(preds)
        df_pred.to_csv(os.path.join(self.model_dir, "multi_property_predictions.csv"), index=False)

        # 分别绘制散点图
        for prop in self.property_columns:
            true_vals = df_pred[prop+"_true"]
            pred_vals = df_pred[prop+"_pred"]
            r2 = r2_score(true_vals, pred_vals)
            mae = mean_absolute_error(true_vals, pred_vals)
            rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
            plt.figure()
            plt.scatter(true_vals, pred_vals, s=80, label=prop)
            plt.text(0.05, 0.85, f"R2: {r2:.2f} MAE: {mae:.2F} RMSE: {rmse:.2f}", transform=plt.gca().transAxes, fontsize=12)
            min_val, max_val = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
            plt.plot([min_val, max_val], [min_val, max_val], "k--")
            plt.xlabel(f"True {prop}")
            plt.ylabel(f"Pred {prop}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f"scatter_{prop}.jpg"), dpi=500)
            plt.close()
              
        
            
if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description = "Running Training Toolkits")

    parser.add_argument("-d", "--debug", action='store_true', default = False,
                        help = "Activating the debug information.")

    parser.add_argument("-s", "--start", action='store_true', default = False,
                        help = "Start training")

    # option for validation
    parser.add_argument("-v", "--validate", action='store_true', default = False,
                        help = "Start validation")

    parser.add_argument("-r", "--restart", action='store_true', default = False,
                        help = "ReStart training from beginning")

    parser.add_argument("--lr", default = 0.01, help = "Specify learning rate", type=float)
    # parser.add_argument("--maxiter", default = 1000, help = "Maximum number of iterations", type=int)

    # parse the arguments from standard input
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.start:

        trainer = Training_Autoencoder()

        if args.restart:
            trainer.load_extractor()
            trainer.load_predictor()
            trainer.load_optimizer(restart=True)

        trainer.load_dataset()
        # trainer.load_extractor()
        trainer.train(lr=args.lr, restart=args.restart)
        # trainer.save_model()
        trainer.infer()

    if args.validate:

        trainer = Training_Autoencoder()

        trainer.load_dataset()
        trainer.load_extractor()
        trainer.load_predictor()
        trainer.infer()


