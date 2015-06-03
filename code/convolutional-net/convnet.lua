require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'

--nn.Parallel,
--nn.Sequential,
--nn.TemporalConvolution,
--nn.SpatialSubSampling,
--nn.Linear



mlp=nn.Parallel(10,1);     -- iterate over dimension 2 of input
mlp:add(nn.Linear(10,3)); -- apply to first slice
mlp:add(nn.Linear(10,2))  -- apply to first second slice

print(torch.randn(10,2))

--print(mlp:forward())
