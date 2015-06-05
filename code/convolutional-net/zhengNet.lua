require 'torch'
require 'nn'

input_length   = 256
n_feature_maps = 10
n_channels     = 3
filter_size    = 5
step           = 2
hidden_units   = 732
n_classes      = 3
n_iterations   = 100

mlp        = nn.Sequential()
main_model = nn.Parallel(2,1)

for i = 1,n_channels do
    local model = nn.Sequential()
    model:add(nn.TemporalConvolution(1,n_feature_maps,filter_size))
    model:add(nn.Sigmoid())
    model:add(nn.TemporalSubSampling(n_feature_maps,2,2))

    model:add(nn.TemporalConvolution(n_feature_maps,n_feature_maps,filter_size))
    model:add(nn.Sigmoid())
    model:add(nn.TemporalSubSampling(n_feature_maps,2,2))
   
    main_model:add(model)
end

x = ((input_length + (1 - filter_size)*(1 +step))/ (step^2))* n_channels

mlp:add(main_model)
mlp:add(nn.Reshape(1,n_feature_maps*x))
mlp:add(nn.Linear(n_feature_maps*x,hidden_units))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(hidden_units,n_classes))

--Lets train
for u=1,n_iterations do
  input = torch.randn(input_length, n_channels, 1)
  y     = torch.ones(3)
  pred  = mlp:forward(input)

  criterion = nn.MSECriterion()
  local err = criterion:forward(pred,y)

  print("error "..err)
  if u == n_iterations then
   print(pred)
  end
  local gradCriterion = criterion:backward(pred,y);
  mlp:zeroGradParameters()
  mlp:backward(input, gradCriterion); 
  mlp:updateParameters(0.01); 
end
