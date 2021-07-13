import os
import random
import time
import argparse
import sys
import numpy

from utils.helpers import FormatterNoDuplicate

def parse_arguments(args_to_parse):

  description = "Used to test various strategies in an automated manner..."
  parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)
  parser.add_argument('strategy',type = str,help = 'Choose one among (sensitive,gray,random)')
  parser.add_argument('--compress',action = 'store_true',help = 'If you want to test the compression case.')
  args = parser.parse_args()
  return args

args = parse_arguments(sys.argv[1:])

random.seed(time.time())

SENSITIVE = False
GRAY = False


COMPRESS = args.compress 

if args.strategy == 'sensitive':
	SENSITIVE = True
elif args.strategy == 'gray':
	GRAY = True

TOTAL_TESTS = 100  # note each test generates a new batch of encyst samples and they are tested on all the attacked models.
# note keep the number of TESTS in gray box to be lesser..it is proportional to len(attack_model)^2


if SENSITIVE:
	file = open("sensitive_results.txt","r+")
elif GRAY:
	file = open("gray_results.txt","r+")
else:
	file = open("random_results.txt","r+")
file.truncate(0)
file.close()

test_num = 0

encyst_cmd_line = "python encyst_samples.py "
sensitive_command_line = "python main_viz.py new_vae --encyst --samples 2 --iter 100 --sensitive --rate 0.0000025 "
gray_command_line = "python main_viz.py new_vae --gaussian --encyst --rate 0.01 --samples 6 --iter 2000 --gray_box "
random_command_line = "python main_viz.py new_vae --gaussian --encyst --rate 0.01 --samples 4 --iter 1000 "

am_paths_list = ["classifers/badnet.pth","classifers/clean_label.pth","classifers/trojannn.pth",
						"classifers/apple_badnet.pth","classifers/apple_trojan.pth","classifers/apple_clean_label.pth",
						"classifers/apple_latent_backdoor.pth","classifers/hidden_trigger.pth"]

seed = random.randint(0,10000)

for test_num in range(TOTAL_TESTS):
	
	print('-----------NEW TEST : '+str(test_num)+' -------------')
	seed = seed + 1

	if SENSITIVE:
		file = open("sensitive_results.txt","a")
		file.write("\n -----------------NEW SENSITIVE WATERMARK GENERATED--------------------------\n")
		file.close()
		os.system(sensitive_command_line+" --seed "+str(seed))

		for attack_name in am_paths_list:
			os.system(encyst_cmd_line+" --sensitive"+" --am_path "+attack_name)
		
	elif GRAY:

		for i in range(len(am_paths_list)):
			
			file = open("gray_results.txt","a")
			file.write("\n -----------------NEW GRAY WATERMARK GENERATED--------------------------\n")
			file.write("\n---The attack model used for generation is : "+am_paths_list[i])
			file.close()

			os.system(gray_command_line+" --seed "+str(seed) +" --am_path "+ am_paths_list[i])
			for j in range(len(am_paths_list)):
				if j!=i:
					os.system(encyst_cmd_line+" --gray_box "+" --am_path "+am_paths_list[j])

	else:
		file = open("random_results.txt","a")
		file.write("\n -----------------NEW RANDOM WATERMARK GENERATED--------------------------\n")
		file.close()

		if COMPRESS:
			os.system(random_command_line+" --seed "+str(seed)+" --compress")
			os.system(encyst_cmd_line+" --compress")
			continue

		os.system(random_command_line+" --seed "+str(seed))

		for attack_name in am_paths_list:
			os.system(encyst_cmd_line+" --am_path "+attack_name)

