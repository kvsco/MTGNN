from data_provider import data_provider
from train_utils import EarlyStopping, visual, metric

from net import gtnet
from torch.optim import lr_scheduler

import numpy as np
import os
import time
import torch
import torch.nn as nn

class Trainer():
    def __init__(self, config):
        self.args = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._model = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, device=self.device, num_nodes=self.args.num_nodes,
              dropout=0.3, subgraph_size=20, node_dim=self.args.node_dim, dilation_exponential=1, conv_channels=32,
              residual_channels=32, skip_channels=64, end_channels=128, seq_length=self.args.seq_len,
              in_dim=1, out_dim=self.args.pred_len, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)

        model_parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Build model, parameter size : {params}")

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self, setting):
        print("Begin training...")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=3, verbose=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        self._model.to(self.device)
        self._model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                # (batch, lookback, node)
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                batch_x = batch_x.permute(0, 2, 1).unsqueeze(1)  # Permute and add channel dimension
                batch_y = batch_y.permute(0, 2, 1).unsqueeze(1)
                # (batch, 1, node, lookback)


                outputs = self._model(batch_x)
                outputs = outputs.view(outputs.size(0), self.args.num_nodes, self.args.pred_len)
                batch_y = batch_y.view(batch_y.size(0), self.args.num_nodes, self.args.pred_len)
                # (batch, node, lookahead )
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self._model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self._model.load_state_dict(torch.load(best_model_path))

        return self._model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self._model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                # (batch, lookback, node)
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                batch_x = batch_x.permute(0, 2, 1).unsqueeze(1)  # Permute and add channel dimension
                batch_y = batch_y.permute(0, 2, 1).unsqueeze(1)
                # (batch, 1, node, lookback)

                outputs = self._model(batch_x)
                outputs = outputs.view(outputs.size(0), self.args.num_nodes, self.args.pred_len)
                batch_y = batch_y.view(batch_y.size(0), self.args.num_nodes, self.args.pred_len)
                # (batch, node, lookahead )

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self._model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self._model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self._model.to(self.device)
        self._model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                # (batch, lookback, node)
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                input = batch_x
                batch_x = batch_x.permute(0, 2, 1).unsqueeze(1)  # Permute and add channel dimension
                batch_y = batch_y.permute(0, 2, 1).unsqueeze(1)
                # (batch, 1, node, lookback)

                outputs = self._model(batch_x)
                outputs = outputs.view(outputs.size(0), self.args.num_nodes, self.args.pred_len)
                batch_y = batch_y.view(batch_y.size(0), self.args.num_nodes, self.args.pred_len)
                # (batch, node, lookahead )
                outputs = outputs.permute(0, 2, 1)
                batch_y = batch_y.permute(0, 2, 1) # (batch, lookahead, nodes)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 10 == 0:
                    input = input.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    visual(gt, pd, self.args.target, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self._model.load_state_dict(torch.load(best_model_path))

        preds = []

        self._model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
