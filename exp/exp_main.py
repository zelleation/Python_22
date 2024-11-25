from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, PatchTST, SparseTSF
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
#mindspore
import mindspore
import mindspore.nn
import mindspore.ops
import mindspore.dataset
import mindspore.amp

from mindspore import serialization
from mindspore import context, Tensor
warnings.filterwarnings('ignore')
def new_adjust_learning_rate(optimizer, scheduler_last_lr,epoch, args, printout=True):
    # 根据epoch和args确定学习率调整策略
    if args.lradj == 'type1':
        lr_adjust = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    elif args.lradj == 'type2':
        lr_adjust_map = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        lr_adjust = lr_adjust_map.get(epoch, args.learning_rate)
    elif args.lradj == 'type3':
        lr_adjust = args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))
    elif args.lradj == 'constant':
        lr_adjust = args.learning_rate
    elif args.lradj == '3':
        lr_adjust = args.learning_rate if epoch < 10 else args.learning_rate * 0.1
    elif args.lradj == '4':
        lr_adjust = args.learning_rate if epoch < 15 else args.learning_rate * 0.1
    elif args.lradj == '5':
        lr_adjust = args.learning_rate if epoch < 25 else args.learning_rate * 0.1
    elif args.lradj == '6':
        lr_adjust = args.learning_rate if epoch < 5 else args.learning_rate * 0.1
    elif args.lradj == 'TST':
        # 假设scheduler_last_lr是一个包含最后学习率的变量
        lr_adjust = scheduler_last_lr
    else:
        lr_adjust = args.learning_rate

    # 更新优化器的学习率
    optimizer.set_lr(lr_adjust)

    if printout:
        print('Updating learning rate to {}'.format(lr_adjust))

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SparseTSF': SparseTSF
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            pass#context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=self.args.device_ids)#model = nn.DataParallel(model, device_ids=self.args.device_ids)
            #它也不支持，咱也用不上
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = mindspore.experimental.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = mindspore.nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = mindspore.nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = mindspore.nn.SmoothL1Loss()
        else:
            criterion = mindspore.nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
#        with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = mindspore.ops.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = mindspore.ops.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                #with torch.cuda.amp.autocast():
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = mindspore.amp.DynamicLossScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        scheduler=mindspore.nn.learning_rate_schedule.ExponentialDecayLR(self.args.learning_rate, 0.9, self.args.train_epochs)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = mindspore.ops.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = mindspore.ops.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    #with torch.cuda.amp.autocast():
                    if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)#为什么可以传入这么多参数？A: 这是因为模型的forward函数定义了这么多参数，没有啊，只有一个输入参数forward(self, x).A: 这是因为forward函数的参数是可变的，可以传入任意多个参数，但是在实际调用时，只传入了一个参数。

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    scaler.minimize(model_optim, scaled_loss)
                else:
                    loss.backward()
                    model_optim.step()
                    


                if self.args.lradj == 'TST':#TST是什么？A: TST是一个调整学习率的方法，可以在训练过程中动态调整学习率
                    global_step = epoch * train_steps + i
                    new_lr=scheduler(global_step)
                    new_adjust_learning_rate(model_optim, new_lr, epoch + 1, self.args, printout=False)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                new_adjust_learning_rate(model_optim, 0, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = os.path.join(path ,'checkpoint.ckpt')#这是什么？.pth是什么文件？A: .pth是pytorch的模型文件，保存了模型的参数，可以用来恢复模型。
        #mindspore中的模型文件是什么？A: .ckpt是mindspore的模型文件，保存了模型的参数，可以用来恢复模型。会自动保存吗？A: 会的，可以通过设置checkpoint保存模型。
        if os.path.exists(best_model_path):
            self.model.set_param_dict(serialization.load_checkpoint(best_model_path))#self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))

        return self.model

    def test(self, setting, test=0):
       
        return

    def predict(self, setting, load=False):
       
        return
