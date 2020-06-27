
import os
result_dir = "../results"
data_dirs  = os.listdir(result_dir)
data_dirs.sort()

for result in data_dirs:
	command ="python holidays_map.py "+result_dir+"/"+result
	os.system(command)

