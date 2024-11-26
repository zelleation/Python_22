import os
# import torch
import mindspore as ms
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):

        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            ms.context.set_context(device_id=self.args.gpu, mode=ms.context.GRAPH_MODE)
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            # CPU，device_id设置为0
            ms.context.set_context(device_id=0, mode=ms.context.GRAPH_MODE, target=ms.context.Target('CPU'))
            print('Use CPU')
        
        # device = ms.get_context('device')
        # 原文那个device在mindspore是没有的，后面涉及到device使用mindspore的set_context设置
        return 

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
