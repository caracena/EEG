require 'torch'
local f = io.open('/home/ubuntu/EEG_data/txt1/s01.txt','r')

local i = 0
for line in f:lines('*l') do
	i = i +1
        local l  = line:split(' ')
end

print(i)
