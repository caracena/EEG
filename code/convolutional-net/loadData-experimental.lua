require 'torch'

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

--toy example
--local filePath = '/home/ubuntu/EEG/code/data.txt'
--real file
local filePath = '/home/ubuntu/EEG_data/txt1/s01.txt'


--Get rows, cols
local nrows = 0 
for line in io.lines(filePath) do
        ncols = #line:split(' ')
        nrows = nrows + 1
end

print(nrows)
print(ncols)

local data     = torch.Tensor(nrows, ncols)

local i = 0
for line in io.lines(filePath) do
	local l = line:split(' ')
	for key, val in ipairs(l) do
		data[i+1][key] = val
   	end
	i = i + 1
end


--comando util para testing: 
--head -n 1 file.txt | cut -c-20
--print(data[1][1])
print(data:size())

--segment:

d = torch.Tensor(data)
print(d:size()) 

trials = d:chunk(40,2)

print(trials[1]:size())


