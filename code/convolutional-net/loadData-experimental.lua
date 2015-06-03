require 'torch'

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

local filePath = '/home/ubuntu/EEG/code/data.txt'
local data     = torch.Tensor(3, 3)

local i = 1
for line in io.lines(filePath) do
	local l = line:split(' ')
	for key, val in ipairs(l) do
		data[i][key] = val
   	end
	i = i + 1
end

print(data)
