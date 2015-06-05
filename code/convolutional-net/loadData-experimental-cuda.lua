require 'torch'
require 'nn'

require 'cunn'
require 'cutorch'

local filePath = '/home/ubuntu/EEG_data/s'
local labelPath = '/home/ubuntu/EEG_labels/labels.csv'

nusers = 1
nchannels = 40
n_feature_maps = 10

-------------------------

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

-------------------------
function loadLabels(path)
 local v = torch.Tensor(40)
 local i = 1 
 for line in io.lines(path) do
  local l = line:split(",")
  for key, val in ipairs(l) do
   v[i] = val
  end
  i=i+1
 end
 return v
end
-------------------------
function getLabelVector(v, i) -- i es el indice del trial, i in (1,40)
 x = torch.Tensor(6):zero()
 class = v[i] -- actualmente retorna class in (3,8) (6 posibles clases) 
 x[class-2] = 1
 return x
end
-------------------------
function loadUserData(path)

 local nrows = 0
 for line in io.lines(path) do
        ncols = #line:split(' ')
        nrows = nrows + 1
 end

 local data = torch.Tensor(nrows, ncols,1)

 local i = 0
 for line in io.lines(path) do
        local l = line:split(' ')
        for key, val in ipairs(l) do
                data[i+1][key] = val
        end
        i = i + 1
 end
 local d = torch.Tensor(data)
 trials  = d:chunk(40,2)
 return trials
end
---------------------------

--input = torch.rand(8046, 40,1)
--print(input:size())

--input = loadUserData('/home/ubuntu/EEG_data/s01.txt')
--print(input[1]:size())

n_feature_maps = 10
mlp = nn.Sequential()

main_model=nn.Parallel(2,1) 
for i = 1,40 do -- using 40 channels
    local model = nn.Sequential()
    model:add(nn.TemporalConvolution(1,n_feature_maps,5))
    model:add(nn.Sigmoid())
    model:add(nn.TemporalSubSampling(n_feature_maps,2,2))

    model:add(nn.TemporalConvolution(n_feature_maps,n_feature_maps,5))
    model:add(nn.Sigmoid())
    model:add(nn.TemporalSubSampling(n_feature_maps,2,2))
   
    main_model:add(model)
end

mlp:add(main_model)
mlp:add(nn.Reshape(1,n_feature_maps*80520))

mlp:add(nn.Linear(n_feature_maps*80520,732))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(732,6))


-- Lets train
labels = loadLabels(labelPath)

for u=1,31 do -- 32 es el numero de usuario,dejo uno afuera
 if u <10 then
  path = filePath .."0"..u..".txt"
 else
  path = filePath..u..".txt"
 end
 print(path)
 input = loadUserData(path)

 for i=1,40 do
  y    = getLabelVector(labels,i)
  pred = mlp:forward(input[i])

  criterion = nn.MSECriterion()
  local err = criterion:forward(pred,y)

  print("error "..err)

  local gradCriterion = criterion:backward(pred,y);
  mlp:zeroGradParameters()
  mlp:backward(input[i], gradCriterion); 
  mlp:updateParameters(0.01);
 end
end
