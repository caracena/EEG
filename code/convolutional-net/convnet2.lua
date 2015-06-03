require 'nn'

nInputSamples = 3
nOutputSamples = 256
nPoints = 256
kernelSize = 5

input = torch.rand(nPoints, nInputSamples)
--print(input)

n_feature_maps = 10


mlp=nn.Parallel(2,1)
for i= 1,3 do
    print(i)
    local model = nn.Sequential()
    model:add(nn.TemporalConvolution(1,n_feature_maps,5))
    model:add(nn.Sigmoid())
    model:add(nn.TemporalSubSampling(n_feature_maps,2,2))

    model:add(nn.TemporalConvolution(n_feature_maps,n_feature_maps,5))
    model:add(nn.Sigmoid())
    model:add(nn.TemporalSubSampling(n_feature_maps,2,2))
    mlp:add(model)
end


output = mlp:forward(input)
