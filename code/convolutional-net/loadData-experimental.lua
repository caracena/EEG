require 'torch'

local filePath = '/home/ubuntu/EEG_data/s'
local labelPath = '/home/ubuntu/EEG_labels/labels.csv'

nusers = 1
-------------------------

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

-- retorna 40 chunks(trials) de (8064x40)
function loadUserData(path)

 local nrows = 0
 for line in io.lines(path) do
        ncols = #line:split(' ')
        nrows = nrows + 1
 end

 local data = torch.Tensor(nrows, ncols)

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


for i = 1,nusers do
 if i <10 then
  print(filePath .."0"..i..".txt" )
 else
  print(filePath..i..".txt" ) 
 end
end 



