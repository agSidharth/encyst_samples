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
  parser.add_argument('--dataset',type = str,default = 'mnist',help = 'the dataset available are (mnist,cifar10,cifar100,face)')
  parser.add_argument('--compress',action = 'store_true',help = 'If you want to test the compression case.')
  parser.add_argument('--num_tests',type = int,default = 10,help = 'Number of experiments needed to be conducted')
  parser.add_argument('--rate',type = float,default = 0.01,help = 'the range of noise added at each step')
  parser.add_argument('--iter',type = int,default = 1000,help = 'the max_iter upto which we are going to check the results..')
  parser.add_argument('--samples',type = int,default = 5,help = 'The watermark images it will try to generate before filtering usually 5 gives a watermark of size 4 on avg')
  parser.add_argument('--disable_gauss',action='store_true',help = 'If you want to use uniform noise')
  parser.add_argument('--multiple',action = 'store_true',help = 'Use multiple on complete latent vector instead of single at a time')
  parser.add_argument('--labels',type = str,default = None,help = 'Use watermark samples only for specific labels')
  parser.add_argument('--scratch',action = 'store_true',help = 'If you want to take clean model as from scratch and attacked model as fine tuned one..')
  args = parser.parse_args()
  return args

args = parse_arguments(sys.argv[1:])

random.seed(time.time())

if args.dataset != 'mnist':
	args.multiple = True

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



encyst_cmd_line = "python encyst_samples.py " 

if args.dataset == 'cifar':
	other_features = other_features + " --dataset cifar "
	encyst_cmd_line = encyst_cmd_line + " --dataset cifar"
	file.write('Using cifar dataset\n')

elif args.dataset == 'cifar100':
	other_features = other_features + " --dataset cifar100 "
	encyst_cmd_line = encyst_cmd_line + " --dataset cifar100 "
	file.write('Using cifar100 dataset\n')

elif args.dataset == 'face':
	other_features = other_features + " --dataset face "
	encyst_cmd_line = encyst_cmd_line + " --dataset face"
	file.write('Using face dataset\n')

	args.compress = True
	COMPRESS = True

elif args.dataset == 'mnist':
	file.write('Using mnist dataset\n')

else :
	sys.exit()


if args.scratch:
		encyst_cmd_line = encyst_cmd_line + " --scratch "
		file.write('For scratch case\n')




file.close()

test_num = 0

common_command_line = "python main_viz.py new_vae --encyst --rate "+str(args.rate)+" --samples "+str(args.samples)+" --iter "+str(args.iter)
sensitive_command_line 	=	common_command_line +other_features+" --sensitive " 
gray_command_line 			=	common_command_line +other_features+" --gray_box "
random_command_line 		=	common_command_line +other_features

if args.dataset=='mnist':
	am_paths_list = ["classifers/badnet.pth","classifers/clean_label.pth","classifers/trojannn.pth",
						"classifers/apple_badnet.pth","classifers/apple_trojan.pth","classifers/apple_clean_label.pth",
						"classifers/apple_latent_backdoor.pth","classifers/hidden_trigger.pth"]
elif(args.dataset=='cifar100'):
	am_paths_list = ["classifers/resnet_badnet100.pth","classifers/resnet_trojannn100.pth","classifers/resnet_clean_label100.pth"]
else:
	am_paths_list = ["classifers/resnet_badnet.pth","classifers/resnet_trojannn.pth","resnet_clean_label.pth"]
seed = random.randint(0,10000)

START_TIME = time.time()

generation_start_time = 0
generation_total_time = 0

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

		generation_start_time = time.time()

		if args.scratch:
			os.system(random_command_line+" --seed "+str(seed)+" --scratch ")

			generation_total_time += time.time() - generation_start_time
			os.system(encyst_cmd_line)
			continue

		if COMPRESS:
			os.system(random_command_line+" --seed "+str(seed)+" --compress")

			generation_total_time += time.time() - generation_start_time
			os.system(encyst_cmd_line+" --compress")
			continue

		
		os.system(random_command_line+" --seed "+str(seed))

		generation_total_time += time.time() - generation_start_time
		for attack_name in am_paths_list:
			os.system(encyst_cmd_line+" --am_path "+attack_name)


END_TIME = time.time()

if GRAY:
	file = open("gray_results.txt","a")
else:
	file = open("random_results.txt","a")

file.write("\nTotal watermark generation time : "+str(generation_total_time))
file.write("\nTotal runtime is : "+str(END_TIME - START_TIME))
file.close()
