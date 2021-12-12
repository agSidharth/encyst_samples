import argparse
import os
import sys

from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from utils.dataset_face import data_loader
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from net.models import LeNet_5
from resnet import ResNet18
from main_viz import WrappedModel
from finetune import ModifiedVGG16Model

def parse_arguments(args_to_parse):

  description = "Filtering and testing of the encyst samples watermark...."
  parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

  parser.add_argument('--dataset',type = str,default = 'mnist',help='The dataset you want to test on..')
  parser.add_argument('-s', '--seed', type=int, default=71,
                        help='Random seed. Can be `None` for stochastic behavior.')
  parser.add_argument('--show', action='store_true',
                        help='Displays the images of the natural samples too...')
  parser.add_argument('--img_size',type = int,default = 64,help='The size of image that we are oging to deal with')
  parser.add_argument('--sensitive',action='store_true', help='If it is sensitive samples..')

  parser.add_argument('--gray_box',action = 'store_true',help = 'If it is gray box model')

  parser.add_argument('--weak_natural',action='store_true',help = 'Weak naturality check parameter')

  parser.add_argument('--min_sens',type = int, default = 1e5, help = 'minimum sensitivity required..')

  parser.add_argument('--arch_path', default='classifers/net_architecture.pth',
                      help='the model architecture path')
  
  parser.add_argument('--mod_path',default='classifers/net.pth',help='the classifier path')

  parser.add_argument('--am_path', default='classifers/square_white_tar0_alpha0.00_mark(3,3).pth',
                      help='the attack model architecture path')

  parser.add_argument('--wm_path', default='results/new_vae/watermark.pth', help='the watermark path')
  parser.add_argument('--compress',action = 'store_true',help = 'If you want to test model compression')
  parser.add_argument('--scratch',action = 'store_true',help = 'If you want to take clean model as from scratch and attacked model as fine tuned one..')

  args = parser.parse_args()
  return args



args = parse_arguments(sys.argv[1:])

SEED = args.seed
SHOW_PLOTS = args.show

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available() and args.dataset!='mnist' and (not args.compress):
  device = 'cuda'
else:
  device = 'cpu'




if args.dataset=="cifar":
  print('For cifar datset.......')
  if args.arch_path=='classifers/net_architecture.pth':
    args.arch_path = 'classifers/resnet_architecture.pth'
  
  if args.mod_path=='classifers/net.pth':
    args.mod_path = 'classifers/resnet18_comp.pth'

elif args.dataset=="cifar100":
  print('For cifar100 datset.......')
  if args.arch_path=='classifers/net_architecture.pth':
    args.arch_path = 'classifers/resnet_architecture100.pth'
  
  if args.mod_path=='classifers/net.pth':
    args.mod_path = 'classifers/resnet18_comp100.pth'


elif args.dataset =='mnist':
  print('For mnist dataset')

elif args.dataset == 'face':
  print('For face dataset......')
  args.arch_path = None

  if args.mod_path == 'classifers/net.pth':
      if args.scratch:
          args.mod_path = 'classifers/clean_face_model_scratch.pth'
          args.am_path = 'classifers/clean_face_model.pth'
      else:
          args.mod_path = 'classifers/clean_face_model.pth'
          args.am_path = 'classifers/prunned_face_model.pth'

else:
  print('Dataset not suported')





if args.sensitive:

  if (args.wm_path).find('sens')==-1:
    args.wm_path = 'results/new_vae/watermark_sens.pth'
    print('\nLoaded the default sensitive model\n')

  else:
    print('\nLoaded your sensitve watermark\n')

if args.gray_box:
  if (args.wm_path).find('gray')==-1:
    args.wm_path = 'results/new_vae/watermark_gray.pth'
    print('\nLoaded the gray box model\n')

  else:
    print('\nLoaded your gray box\n')

samples_per_dim = 10

if args.dataset == 'mnist':
  img_size = args.img_size
  transforms_1 = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor()])


  mnist_trainset_i = datasets.MNIST(root='./data', train=True, download=True, transform=transforms_1)
  trainset = DataLoader(mnist_trainset_i,batch_size = 100,shuffle=True)
  num_classes = 10

  natural_samples = torch.zeros(num_classes,samples_per_dim,1,img_size,img_size)      #10 classes , 10 samples per dimension...

elif args.dataset == 'cifar':
  img_size = 32
  transforms_1 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

  cifar_trainset = datasets.CIFAR10(root='./data',train=True,download=True,transform=transforms_1)
  trainset = DataLoader(cifar_trainset,batch_size = 100,shuffle = True)
  num_classes = 10

  natural_samples = torch.zeros(num_classes,samples_per_dim,3,img_size,img_size)      #10 classes , 10 samples per dimension...

elif args.dataset == "cifar100":

  img_size = 64
  transforms_1 = transforms.Compose([transforms.Resize(img_size),
            transforms.CenterCrop(img_size),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

  cifar_trainset = datasets.CIFAR100(root='./data',train=True,download=True,transform=transforms_1)
  trainset = DataLoader(cifar_trainset,batch_size = 100,shuffle = True)
  num_classes = 100

  natural_samples = torch.zeros(num_classes,samples_per_dim,3,img_size,img_size)


elif args.dataset == 'face':
  img_size = 224
  num_classes = 10

  transforms_1 = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  trainset,_ = data_loader(transforms_1,100,0,1000)

  natural_samples = torch.zeros(num_classes,samples_per_dim,3,img_size,img_size)      #10 classes , 10 samples per dimension...
  #sys.exit()

completed = [0]*num_classes
total = 0

for inputs,labels in trainset:
  for (idx,label) in enumerate(labels):

    if (completed[label.numpy()] <= samples_per_dim - 1):
      natural_samples[label.numpy()][completed[label.numpy()]] = inputs[idx]
      completed[label.numpy()] += 1
      total += 1
    
    if (total>=num_classes*samples_per_dim):
      break
  if (total>=num_classes*samples_per_dim):
    break

#plt.imshow(np.transpose(natural_samples[num_classes-1][samples_per_dim-1].detach().numpy(), (1, 2, 0))[:,:,0])

loss_fn_alex = lpips.LPIPS(net='alex')
avg_loss = [0]*num_classes
max_loss = [0]*num_classes

torch.set_grad_enabled(False)

for dim in range(num_classes):
  for i in range(samples_per_dim - 1):
    first_img = natural_samples[dim][i].repeat(3,1,1).unsqueeze_(0) if args.dataset=='mnist' else natural_samples[dim][i].unsqueeze_(0)
    for j in range(i+1,samples_per_dim):
      second_img = natural_samples[dim][j].repeat(3,1,1).unsqueeze_(0) if args.dataset=='mnist' else natural_samples[dim][j].unsqueeze_(0)
      this_loss = loss_fn_alex(first_img, second_img)[0][0][0][0].numpy()
      #print(this_loss)
      avg_loss[dim] += this_loss
      if this_loss>max_loss[dim]:
        max_loss[dim] = this_loss + 0
  avg_loss[dim] = avg_loss[dim]/(int(samples_per_dim*(samples_per_dim-1)/2))

torch.set_grad_enabled(True)

def check_naturality(dimension,img,natural_samples,samples_per_dim):
  max = 0
  sum = 0
  
  first_img = img[0].repeat(3,1,1).unsqueeze_(0) if args.dataset=='mnist' else img[0].unsqueeze_(0)
  for idx in range(samples_per_dim):
    second_img = natural_samples[dimension][idx].repeat(3,1,1).unsqueeze_(0) if args.dataset=='mnist' else natural_samples[dimension][idx].unsqueeze_(0)
    this_loss = loss_fn_alex(first_img, second_img)[0][0][0][0].detach().numpy()

    sum = sum + this_loss
    if this_loss>max:
      max = this_loss
  
  return (max,sum/samples_per_dim)


#"""
if not args.compress and not args.scratch:
  PATH = args.arch_path
  clean_model = torch.load(PATH,map_location=device)
  attacked_model = torch.load(PATH,map_location=device)

  PATH = args.mod_path
  clean_model.load_state_dict(torch.load(PATH,map_location=device))
  clean_model.eval()

  PATH = args.am_path
  attacked_model.load_state_dict(torch.load(PATH,map_location=device))
  attacked_model.eval()

else:
  if args.dataset == 'mnist':
    clean_model = LeNet_5(mask=True)
    clean_model.load_state_dict(torch.load('compression_clfs/net2.pth',map_location='cpu'))
    clean_model.eval()

    attacked_model = LeNet_5(mask=True)
    attacked_model.load_state_dict(torch.load('compression_clfs/model_compressed.pth',map_location='cpu'))
    attacked_model.eval()

    args.mod_path = 'compression_clfs/net2.pth'
    args.am_path = 'compression_clfs/model_compressed.pth'

  elif args.dataset == 'cifar':
    clean_model = WrappedModel(ResNet18())
    clean_model.load_state_dict(torch.load('compression_clfs/resnet2.pth',map_location='cpu')['net'])
    clean_model.eval()

    attacked_model = WrappedModel(ResNet18())
    attacked_model.load_state_dict(torch.load('compression_clfs/resnet_compressed.pth',map_location='cpu')['net'])
    attacked_model.eval()

    args.mod_path = 'compression_clfs/resnet2.pth'
    args.am_path = 'compression_clfs/resnet_compressed.pth'

  else:
    clean_model = torch.load(args.mod_path,map_location = device)
    clean_model.eval()

    attacked_model = torch.load(args.am_path,map_location = device)
    attacked_model.eval()

    

PATH = args.wm_path
watermark = torch.load(PATH,map_location=torch.device('cpu'))

print('\nLoading the watermark at : '+ args.wm_path)

latent_dim = len(watermark["inner_img"])
samples_per_dim = watermark["inner_img"][0].shape[0]


topk = 2                                #NOTE: THIS CAN BE CHANGED.....


prob_similar = 0
k_label_similar = 0
top_label_similar = 0
total = 0
zeros = torch.zeros(1,1,img_size,img_size) if args.dataset=='mnist' else torch.zeros(1,3,img_size,img_size)
zero_num = 0

TOTAL_WM_IMGS = 0

print('\nTHE DIMENSION NUMBER ARE DONE INDEX WISE NOT ACTUAL LATENT CODE DIMENSION\n')

print('\n\n')
for dim in range(latent_dim):
  for sample in range(samples_per_dim):

    if not args.gray_box:
      ###### For the inner image
      inner_img = watermark["inner_img"][dim][sample]

      if (not torch.equal(inner_img,zeros)):

        TOTAL_WM_IMGS += 1

        clean_output = clean_model((inner_img).to(device)).data
        clean_topk   = torch.topk(clean_output,topk).indices
        _, clean_pred = torch.max(clean_output, 1)
        
        

        max_sample,avg_sample = check_naturality(watermark["inner_pred"][dim][sample].numpy(),inner_img,natural_samples,samples_per_dim)

        check_sensitivity = True

        if args.sensitive:
          
          if(watermark["inner_sens"][dim][sample]<args.min_sens):
            check_sensitivity = False

        if not args.weak_natural:
          check_sensitivity = avg_sample<avg_loss[dim] and check_sensitivity

        if (max_sample< max_loss[dim] and check_sensitivity):

          if args.sensitive:
            print('The sensitivity of inner image is : '+str(watermark["inner_sens"][dim][sample]))
          print(" Inner image : dim : "+str(dim)+' ,clean label : '+str(clean_pred))
              
          if(SHOW_PLOTS):   
            plt.figure(figsize=(2,2))
            display_img = np.transpose(inner_img[0].detach().numpy(), (1, 2, 0))
            plt.imshow(display_img[:,:,0] if args.dataset=='mnist' else display_img)
            plt.show()

          if args.dataset == 'mnist':
            torchvision.utils.save_image(inner_img, 'mnist_images/'+str(dim)+':'+str(sample)+'inner.png', normalize=True, range=(-1, 1))
          elif args.dataset == 'cifar':
            torchvision.utils.save_image(inner_img, 'cifar_images/'+str(dim)+':'+str(sample)+'inner.png', normalize=True, range=(-1, 1))
          elif args.dataset == 'cifar100':
            torchvision.utils.save_image(inner_img, 'cifar100_images/'+str(dim)+':'+str(sample)+'inner.png', normalize=True, range=(-1, 1))


          attacked_output = attacked_model((inner_img).to(device)).data
          attacked_topk   = torch.topk(attacked_output,topk).indices
          _, attacked_pred = torch.max(attacked_output, 1)
          print('Attacked label : '+str(attacked_pred))
          
          total = total + 1

          if (torch.equal(clean_output,attacked_output)):
            prob_similar = prob_similar + 1
            print("The inner image prob vector matched")
            
          if (torch.equal(clean_topk,attacked_topk)):
            k_label_similar = k_label_similar + 1
            print("The inner image top "+str(topk)+" labels matched")

          if (torch.equal(clean_pred,attacked_pred)):
            top_label_similar = top_label_similar + 1
            print("The inner image top 1 label matched")
          
          print('\n')
        
        else:
          
          #print('The image is not natural enough..')
          zero_num = zero_num + 1
      else:
        
        #print("The image is null, Ignoring this image.....")
        zero_num = zero_num+1
      
      del inner_img


    ###### For the outer image
    
    outer_img = watermark["outer_img"][dim][sample]

    if (not torch.equal(zeros,outer_img)):

      TOTAL_WM_IMGS += 1

      clean_output = clean_model((outer_img).to(device)).data
      clean_topk   = torch.topk(clean_output,topk).indices
      _, clean_pred = torch.max(clean_output, 1)

      

      max_sample,avg_sample = check_naturality(watermark["outer_pred"][dim][sample].numpy(),outer_img,natural_samples,samples_per_dim)

      check_sensitivity = True

      if args.sensitive:
        
        if(watermark["outer_sens"][dim][sample]<args.min_sens):
          check_sensitivity = False

      if not args.weak_natural:
        check_sensitivity = avg_sample<avg_loss[dim] and check_sensitivity

      if (max_sample< max_loss[dim] and check_sensitivity):

        if args.sensitive:
          print('The sensitivity of outer image is : '+str(watermark["outer_sens"][dim][sample]))
        print("Outer image dim : "+str(dim)+' , clean label : ' +str(clean_pred))

        if(SHOW_PLOTS):   
          plt.figure(figsize=(2,2))
          display_img = np.transpose(outer_img[0].detach().numpy(), (1, 2, 0))
          plt.imshow(display_img[:,:,0] if args.dataset=='mnist' else display_img)
          plt.show()

        if args.dataset == 'mnist':
          torchvision.utils.save_image(outer_img, 'mnist_images/'+str(dim)+':'+str(sample)+'outer.png', normalize=True, range=(-1, 1))
        elif args.dataset == 'cifar':
          torchvision.utils.save_image(outer_img, 'cifar_images/'+str(dim)+':'+str(sample)+'outer.png', normalize=True, range=(-1, 1))
        elif args.dataset == 'cifar100':
          torchvision.utils.save_image(outer_img, 'cifar100_images/'+str(dim)+':'+str(sample)+'outer.png', normalize=True, range=(-1, 1))

        attacked_output = attacked_model((outer_img).to(device)).data
        attacked_topk   = torch.topk(attacked_output,topk).indices
        _, attacked_pred = torch.max(attacked_output, 1)
        print('Attacked label : '+str(attacked_pred))

        total = total + 1
        if (torch.equal(clean_output,attacked_output)):
          prob_similar = prob_similar + 1
          print("The outer image prob vector matched")
          
        if (torch.equal(clean_topk,attacked_topk)):
          k_label_similar = k_label_similar + 1
          print("The outer image top "+str(topk)+" labels matched")

        if (torch.equal(clean_pred,attacked_pred)):
          top_label_similar = top_label_similar + 1
          print("The outer image top 1 label matched")
        
        print('\n')
      
      else:

        #print('The image is not natural enough..')
        zero_num = zero_num + 1
    else:

      #print("The image is null, Igonoring this image....")
      zero_num = zero_num + 1

if(total==0):

  if args.sensitive:
    file = open("sensitive_results.txt","a")
    file.write("\nSensitive")

  elif args.gray_box:
    file = open("gray_results.txt","a")
    file.write("\nGray")

  else:
    file = open("random_results.txt","a")
    file.write("\nRandom")
    
  file.write("\nTotal watermark images : "+str(TOTAL_WM_IMGS))
  file.write('\nSorry no image is natural enough or sensitive enough if required\n')
  file.close()
  sys.exit()

print('The total number of non-zero valid images are : '+str(total)+'/'+str(total+zero_num))
print("The percentage of similar top "+str(topk)+" labels is "+str(k_label_similar/total*100))
print('The percentage of similar top label is '+str(top_label_similar/total*100))
print('\n')

if args.sensitive:
  file = open("sensitive_results.txt","a")
  file.write("\nSensitive")

elif args.gray_box:
  file = open("gray_results.txt","a")
  file.write("\nGray")

else:
  file = open("random_results.txt","a")
  file.write("\nRandom")

  if args.compress:
    file.write("\nCompression case")



if args.dataset=='mnist':
  file.write('--------mnist\n')
elif args.dataset=='cifar':
  file.write('--------cifar\n')
elif args.dataset=='cifar100':
  file.write('--------cifar100\n')
else :
  file.write('--------face\n')

file.write("\nTotal watermark images : "+str(TOTAL_WM_IMGS))
file.write("\nTotal natural watermark images : "+str(total))
file.write("\nThe percentage of similar top "+str(topk)+" labels is "+str(k_label_similar/total*100))
file.write('\nThe percentage of similar top label is '+str(top_label_similar/total*100))

if top_label_similar==total:
  file.write('\nWatermark of size '+str(total)+' FAILED for '+args.am_path+" \n")
else:
  file.write('\nWatermark of size '+str(total)+' SUCCESS for '+args.am_path+" \n")

file.close()