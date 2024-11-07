import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.regression import *
from ignite.utils import *

from ..utils import convert_to_tensor
from ..utils import CustomDataset2


class SSIM_metric(nn.Module):
	def __init__(self, device='cpu'):
		super(SSIM_metric, self).__init__()

		def eval_step(engine, batch):
			return batch
		self.default_evaluator = Engine(eval_step)
		self.device = torch.device(device)
	
	def forward(self, xs, ys, batch_size=1024, data_range=255):
		metric = SSIM(device=self.device, data_range=data_range)
		metric.attach(self.default_evaluator, "ssim")

		xs = convert_to_tensor(xs)
		ys = convert_to_tensor(ys)

		xs_min = torch.min(xs, dim=1, keepdim=True)[0]
		xs_max = torch.max(xs, dim=1, keepdim=True)[0]
		xs = data_range * (xs - xs_min) / (xs_max - xs_min)

		ys_min = torch.min(ys, dim=1, keepdim=True)[0]
		ys_max = torch.max(ys, dim=1, keepdim=True)[0]
		ys = data_range * (ys - ys_min) / (ys_max - ys_min)

		xs = xs.reshape(xs.shape[0], xs.shape[1], 1, 1)
		ys = ys.reshape(ys.shape[0], ys.shape[1], 1, 1)
		dataset = CustomDataset2(xs, ys)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

		state = self.default_evaluator.run(dataloader, max_epochs=1)
		return state.metrics["ssim"]
	

