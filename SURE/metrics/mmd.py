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


class MMD_metric(nn.Module):
	def __init__(self, var=1.0, device='cpu'):
		super(MMD_metric, self).__init__()

		def eval_step(engine, batch):
			return batch
		self.default_evaluator = Engine(eval_step)
		self.device = torch.device(device)
		self.var = var
	
	def forward(self, xs, ys, batch_size=1024):
		metric = MaximumMeanDiscrepancy(var=self.var, device=self.device)
		metric.attach(self.default_evaluator, "mmd")

		xs = convert_to_tensor(xs)
		ys = convert_to_tensor(ys)
		dataset = CustomDataset2(xs, ys)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

		state = self.default_evaluator.run(dataloader, max_epochs=1)
		return state.metrics["mmd"]
	

