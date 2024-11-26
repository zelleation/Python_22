import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
def test(self, setting, test=0):
   test_data, test_loader = self._get_data(flag='test')

     if test:
         print('loading model')
         # 加载模型参数
         param_dict = load_checkpoint(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
         load_param_into_net(self.model, param_dict)

     preds = []
     trues = []
     inputx = []
     folder_path = './test_results/' + setting + '/'
     if not os.path.exists(folder_path):
         os.makedirs(folder_path)

     # 将模型设置为评估模式（MindSpore中对应为set_train(False)）
     self.model.set_train(False)
     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        # 将数据转换为MindSpore的Tensor类型，并指定数据类型为float32
         batch_x = Tensor(batch_x, dtype=mindspore.float32)
         batch_y = Tensor(batch_y, dtype=mindspore.float32)
         batch_x_mark = Tensor(batch_x_mark, dtype=mindspore.float32)
          batch_y_mark = Tensor(batch_y_mark, dtype=mindspore.float32)

         # decoder input
          # 使用MindSpore的ops.Zeros创建全零张量，形状与batch_y的后一部分相同
          dec_inp = ops.Zeros()((batch_y.shape[0], self.args.pred_len, batch_y.shape[2]), mindspore.float32)
          # 使用MindSpore的ops.Concat在维度1上拼接batch_y的前一部分和dec_inp
          dec_inp = ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])

          # encoder - decoder
          if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
            outputs = self.model(batch_x)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        # 截取输出和真实值的后pred_len部分，并根据f_dim选择维度
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        # 将输出和真实值转换为numpy数组
        pred = outputs.asnumpy()
        true = batch_y.asnumpy()

        preds.append(pred)
        trues.append(true)
        inputx.append(batch_x.asnumpy())
        if i % 20 == 0:
            input = batch_x.asnumpy()
            gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    if self.args.test_flop:
        test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
        exit()

    # fix bug
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
    f = open("result.txt", 'a')
    f.write(setting + "  \n")
    f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
    f.write('\n')
    f.write('\n')
    f.close()

    # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
    # np.save(folder_path + 'pred.npy', preds)
    # np.save(folder_path + 'true.npy', trues)
    # np.save(folder_path + 'x.npy', inputx)
    return

def predict(self, setting, load=False):
    pred_data, pred_loader = self._get_data(flag='pred')

    if load:
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        # 加载模型参数
        param_dict = load_checkpoint(best_model_path)
        load_param_into_net(self.model, param_dict)

    preds = []

    # 将模型设置为评估模式
    self.model.set_train(False)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
        batch_x = Tensor(batch_x, dtype=mindspore.float32)
        batch_y = Tensor(batch_y, dtype=mindspore.float32)
        batch_x_mark = Tensor(batch_x_mark, dtype=mindspore.float32)
        batch_y_mark = Tensor(batch_y_mark, dtype=mindspore.float32)

        # decoder input
        dec_inp = ops.Zeros()((batch_y.shape[0], self.args.pred_len, batch_y.shape[2]), mindspore.float32)
        dec_inp = ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])

        # encoder - decoder
        if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
            outputs = self.model(batch_x)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # 将输出转换为numpy数组
        pred = outputs.asnumpy()
        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'real_prediction.npy', preds)

    return
