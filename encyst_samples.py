import argparse
import os
import sys

from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
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

def parse_arguments(args_to_parse):

  description = "Filtering and testing of the encyst samples watermark...."
  parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

  parser.add_argument('-s', '--seed', type=int, default=71,
                        help='Random seed. Can be `None` for stochastic behavior.')
  parser.add_argument('--show', action='store_true',
                        help='Displays the images of the natural samples too...')
  parser.add_argument('--sensitive',action='store_true', help='If it is sensitive samples..')

  parser.add_argument('--gray_box',action = 'store_true',help = 'If it is gray box model')

  parser.add_argument('--weak_natural',action='store_true',help = 'Weak naturality check parameter')

  parser.add_argument('--arch_path', default='classifers/net_architecture.pth',
                      help='the model architecture path')
  parser.add_argument('--min_sens',type = int, default = 1e5, help = 'minimum sensitivity required..')

  parser.add_argument('--mod_path',default='classifers/net.pth',help='the classifier path')

  parser.add_argument('--attack_mod_path', default='classifers/square_white_tar0_alpha0.00_mark(3,3).pth',
                      help='the attack model architecture path')

  parser.add_argument('--wm_path', default='results/factor_mnist/watermark.pth', help='the watermark path')

  args = parser.parse_args()
  return args



args = parse_arguments(sys.argv[1:])

SEED = args.seed
SHOW_PLOTS = args.show

torch.manual_seed(SEED)
np.random.seed(SEED)

if args.sensitive:

  if (args.wm_path).find('sens')==-1:
    args.wm_path = 'results/factor_mnist/watermark_sens.pth'
    print('\nLoaded the default sensitive model\n')

  else:
    print('\nLoaded your sensitve watermark\n')

if args.gray_box:
  if (args.wm_path).find('gray')==-1:
    args.wm_path = 'results/factor_mnist/watermark_gray.pth'
    print('\nLoaded the gray box model\n')

  else:
    print('\nLoaded your gray box\n')

img_size = 32
transforms_1 = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])


mnist_trainset_i = datasets.MNIST(root='./data', train=True, download=True, transform=transforms_1)
trainset = DataLoader(mnist_trainset_i,batch_size = 100,shuffle=True)


num_classes = 10

samples_per_dim = 10

natural_samples = torch.zeros(num_classes,samples_per_dim,1,img_size,img_size)      #10 classes , 10 samples per dimension...
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
    first_img = natural_samples[dim][i].repeat(3,1,1).unsqueeze_(0)
    for j in range(i+1,samples_per_dim):
      second_img = natural_samples[dim][j].repeat(3,1,1).unsqueeze_(0)
      this_loss = loss_fn_alex(first_img, second_img)[0][0][0][0].numpy()
      #print(this_loss)
      avg_loss[dim] += this_loss
      if this_loss>max_loss[dim]:
        max_loss[dim] = this_loss + 0
  avg_loss[dim] = avg_loss[dim]/(int(samples_per_dim*(samples_per_dim-1)/2))

#print(max_loss)
#print(avg_loss)
torch.set_grad_enabled(True)

def check_naturality(dimension,img,natural_samples,samples_per_dim):
  max = 0
  sum = 0
  
  first_img = img[0].repeat(3,1,1).unsqueeze_(0)
  for idx in range(samples_per_dim):
    second_img = natural_samples[dimension][idx].repeat(3,1,1).unsqueeze_(0)
    this_loss = loss_fn_alex(first_img, second_img)[0][0][0][0].detach().numpy()

    sum = sum + this_loss
    if this_loss>max:
      max = this_loss
  
  return (max,sum/samples_per_dim)


#"""
PATH = args.arch_path
clean_model = torch.load(PATH,map_location=torch.device('cpu'))
attacked_model = torch.load(PATH,map_location=torch.device('cpu'))

PATH = args.mod_path
clean_model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
clean_model.eval()

PATH = args.attack_mod_path
attacked_model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
attacked_model.eval()

PATH = args.wm_path
watermark = torch.load(PATH,map_location=torch.device('cpu'))

print('\nLoading the watermark at : '+ args.wm_path)

latent_dim = len(watermark["inner_img"])
samples_per_dim = watermark["inner_img"][0].shape[0]


"""
clean_model = models.resnet18(pretrained=False)
clean_model.fc = nn.Linear(512,num_classes)
clean_model.conv1 = nn.Conv2d(1, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
PATH = "classifers/mnist_resnet18.pth"
clean_model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
clean_model.eval()

attacked_model = models.resnet18(pretrained=False)
attacked_model.fc = nn.Linear(512,num_classes)
attacked_model.conv1 = nn.Conv2d(1, 32,kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

PATH = "classifers/square_white_tar0_alpha0.00_mark(3,3).pth"
model_dict = torch.load(PATH,map_location=torch.device('cpu'))
#for key in list(model_dict.keys()):
#    model_dict[(key[9:] if 'feature' in key else key[11:])] = model_dict.pop(key)
print(model_dict['classifier.fc1.weight'].shape)
#print(attacked_model)
attacked_model.load_state_dict(model_dict)
attacked_model.eval()

PATH = "results/factor_mnist/watermark.pth"
watermark = torch.load(PATH,map_location=torch.device('cpu'))

latent_dim = len(watermark["inner_img"])
samples_per_dim = watermark["inner_img"][0].shape[0]
"""


topk = 2                                #NOTE: THIS CAN BE CHANGED.....


prob_similar = 0
k_label_similar = 0
top_label_similar = 0
total = 0
zeros = torch.zeros(1,1,img_size,img_size)
zero_num = 0


print('\n\n')
for dim in range(latent_dim):
  for sample in range(samples_per_dim):

    if not args.gray_box:
      ###### For the inner image
      inner_img = watermark["inner_img"][dim][sample]

      if (not torch.equal(inner_img,zeros)):

        clean_output = clean_model((inner_img)).data
        clean_topk   = torch.topk(clean_output,topk).indices
        _, clean_pred = torch.max(clean_output, 1)
        
        

        max_sample,avg_sample = check_naturality(clean_pred[0].detach().numpy(),inner_img,natural_samples,samples_per_dim)

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
            plt.imshow(np.transpose(inner_img[0].detach().numpy(), (1, 2, 0))[:,:,0])
            plt.show()

          attacked_output = attacked_model((inner_img)).data
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

      clean_output = clean_model((outer_img)).data
      clean_topk   = torch.topk(clean_output,topk).indices
      _, clean_pred = torch.max(clean_output, 1)

      

      max_sample,avg_sample = check_naturality(clean_pred[0].detach().numpy(),outer_img,natural_samples,samples_per_dim)

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
          plt.imshow(np.transpose(outer_img[0].detach().numpy(), (1, 2, 0))[:,:,0])
          plt.show()

        attacked_output = attacked_model((outer_img)).data
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


print('The total number of non-zero valid images are : '+str(total)+'/'+str(total+zero_num))
print('The percentage of similar probability vector is '+str(prob_similar/total*100))
print("The percentage of similar top "+str(topk)+" labels is "+str(k_label_similar/total*100))
print('The percentage of similar top label is '+str(top_label_similar/total*100))
