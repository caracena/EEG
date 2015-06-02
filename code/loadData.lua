require 'torch'

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end


local filePath = 'EEG_data/txt1/s01.txt'

-- Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(' ')
  end
  i = i + 1
end
print(i)
print(COLS)
local ROWS = i 

-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
local header = csvFile:read()

local data = torch.Tensor(ROWS, COLS)

local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(' ')
  for key, val in ipairs(l) do
    data[i][key] = val	
  end
end 

print(data[1])

