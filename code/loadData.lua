require 'torch'

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

local filePath = '/home/ubuntu/EEG/code/data.txt'
local data = torch.Tensor(3, 3)

-- Count number of rows and columns in file
local i = 1
for line in io.lines(filePath) do
   --COLS = #line:split(' ')
   local l = line:split(' ')
   for key, val in ipairs(l) do
    data[i][key] = val
    print(val)
   end  
  i = i + 1
end
--print("-----")
--print(i)
--print(COLS)
--print("-----")
--local ROWS = i 

-- Read data from CSV to tensor
--local csvFile = io.open(filePath, 'r')
--local header = csvFile:read()

--local data = torch.Tensor(ROWS, COLS)

--local i = 0
--for line in csvFile:lines('*l') do
  --i = i + 1
  --local l = line:split(' ')
  --print(l) 
  --for key, val in ipairs(l) do
    --data[i][key] = val
    --print(val)	
  --end
  --i = i +1
--end 
--print("------")  
print(data)
print(data[1])
