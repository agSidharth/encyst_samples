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
  parser.add_argument('--dataset',type = str,default = 'mnist',help = 'the dataset available are (mnist,cifar10)')
  parser.add_argument('--compress',action = 'store_true',help = 'If you want to test the compression case.')
  parser.add_argument('--num_tests',type = int,default = 10,help = 'Number of experiments needed to be conducted')
  parser.add_argument('--rate',type = int,default = 0.01,help = 'the range of noise added at each step')
  parser.add_argument('--iter',type = int,default = 1000,help = 'the max_iter upto which we are going to check the results..')
  parser.add_argument('--samples',type = int,default = 5,help = 'The watermark images it will try to generate before filtering usually 5 gives a watermark of size 4 on avg')
  parser.add_argument('--disable_gauss',action='store_true',help = 'If you want to use uniform noise')
  parser.add_argument('--multiple',action = 'store_true',help = 'Use multiple on complete latent vector instead of single at a time')
  parser.add_argument('--labels',type = str,default = None,help = 'Use watermark samples only for specific labels')
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

TOTAL_TESTS = args.num_tests  # note each test generates a new batch of encyst samples and they are tested on all the attacked models.
# note keep the number of TESTS in gray box to be lesser..it is proportional to len(attack_model)^2


if SENSITIVE:
	file = open("sensitive_results.txt","r+")
elif GRAY:
	file = open("gray_results.txt","r+")
else:
	file = open("random_results.txt","r+")
file.truncate(0)

file.write('EXPERIMENT SETTINGS :\n')

file.write('The rate used in this file is : '+str(args.rate)+'\n')
file.write('The max iter checked in this file is : '+str(args.iter)+'\n')

other_features = ""

if not args.disable_gauss:
	other_features = other_features + " --gaussian "
	file.write('Using gaussian noise\n')
else:
	file.write('Using uniform noise\n')

if args.multiple:
	other_features = other_features + " --multiple "
	file.write('Using noise over complete latent vector\n')
else:
	file.write('Using noise per latent dim \n')

if args.labels != None:
	other_features = other_features + " --labels "+ args.labels+" "
	file.write('Using only specific labels which are : '+args.labels+"\n")
else:
	file.write('Using all the labels for watermark samples\n')

if args.dataset == 'cifar':
	other_features = other_features + " --dataset cifar "
	file.write('Using cifar dataset\n')
else:
	file.write('Using mnist dataset\n')

file.close()

test_num = 0

encyst_cmd_line = "python encyst_samples.py " + ("" if args.dataset=="mnist" else " --dataset cifar ")
common_command_line = "python main_viz.py new_vae --encyst --rate "+str(args.rate)+" --samples "+str(args.samples)+" --iter "+str(args.iter)
sensitive_command_line 	=	common_command_line +other_features+" --sensitive " 
gray_command_line 			=	common_command_line +other_features+" --gray_box "
random_command_line 		=	common_command_line +other_features

if args.dataset=='mnist':
	am_paths_list = ["classifers/badnet.pth","classifers/clean_label.pth","classifers/trojannn.pth",
						"classifers/apple_badnet.pth","classifers/apple_trojan.pth","classifers/apple_clean_label.pth",
						"classifers/apple_latent_backdoor.pth","classifers/hidden_trigger.pth"]
else:
	am_paths_list = ["classifers/resnet_badnet.pth","classifers/resnet_trojannn.pth"]
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

