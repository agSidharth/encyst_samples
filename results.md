# Tables for paper:

## Random noise:

### General
EXPERIMENT SETTINGS :  
The rate used in this file is : 0.01  
The max iter checked in this file is : 1000  
Using gaussian noise  
Using noise per latent dim  
Using all the labels for watermark samples  

Watermark size| SUCCESS | TOTAL | SUCCESS RATE|
| ----------- | ------- | ----- | ----------- |
|1|11|24||
|2|277|283||
|3|54|54||
|4|450|450||
|5|80|80||
|6|338|338||
|7|30|30||
|8|155|155||
|9|16|16||
|10|62|62||

### 3 label only  
EXPERIMENT SETTINGS :  
The rate used in this file is : 0.01  
The max iter checked in this file is : 1000  
Using gaussian noise  
Using noise per latent dim  
Using only specific labels which are : 1,4,7  

Watermark size| SUCCESS | TOTAL | SUCCESS RATE|
| ----------- | ------- | ----- | ----------- |
|1|0|0||
|2|331|334||
|3|8|8||
|4|223|224||
|5|0|0||
|6|72|72||
|7|0|0||
|8|8|8||
|9|0|0||
|10|8|8||


### 1 label only
EXPERIMENT SETTINGS :  
The rate used in this file is : 0.01  
The max iter checked in this file is : 1000  
Using gaussian noise  
Using noise per latent dim  
Using only specific labels which are : 4  

Watermark size| SUCCESS | TOTAL | SUCCESS RATE|
| ----------- | ------- | ----- | ----------- |
|1|0|0||
|2|64|64||
|3|0|0||
|4|120|120||
|5|0|0||
|6|24|24||
|7|0|0||
|8|8|8||
|9|0|0||
|10|0|0||

### Increasing noise added at each step from 0.01 to 0.5 i.e rate
EXPERIMENT SETTINGS :  
The rate used in this file is : 0.5  
The max iter checked in this file is : 150
Using gaussian noise  
Using noise per latent dim  
Using all the labels for watermark samples  
Using mnist dataset

Watermark size| SUCCESS | TOTAL | SUCCESS RATE|
| ----------- | ------- | ----- | ----------- |
|1|15|48||
|2|122|240||
|3|47|56||
|4|102|128||
|5|44|64||
|6|56|72||
|7|32|32||
|8||||
|9||||
|10||||


### For noise in mutliple dimensions.

EXPERIMENT SETTINGS :  
The rate used in this file is : 0.01  
The max iter checked in this file is : 800  
Using gaussian noise  
Using noise over complete latent vector  
Using all the labels for watermark samples  
Using mnist dataset  

Watermark size| SUCCESS | TOTAL | SUCCESS RATE|
| ----------- | ------- | ----- | ----------- |
|1|8|16||
|2|279|326||
|3|8|8||
|4|247|248||
|5|8|8||
|6|128|128||
|7|16|16||
|8|32|32||
|9||||
|10|24|24||

### For CIFAR10 compressed attack 

EXPERIMENT SETTINGS :  
The rate used in this file is : 0.02  
The max iter checked in this file is : 800  
Using gaussian noise  
Using noise over complete latent vector  
Using all the labels for watermark samples  
Using cifar dataset  

Watermark size| SUCCESS | TOTAL | SUCCESS RATE|
| ----------- | ------- | ----- | ----------- |
|1|3|3||
|2|35|38||
|3|10|10||
|4|37|37||
|5|12|12||
|6|36|36||
|7|15|15||
|8|11|11||
|9|9|9||
|10|20|20||

### For face dataset.....

EXPERIMENT SETTINGS :    
The rate used in this file is : 0.025    
The max iter checked in this file is : 200    
Using gaussian noise    
Using noise over complete latent vector    
Using all the labels for watermark samples    
Using face dataset    

Watermark size| SUCCESS | TOTAL | SUCCESS RATE|
| ----------- | ------- | ----- | ----------- |
|1||||
|2|53|54||
|3||||
|4|29|29||
|5|1|1||
|6|13|13||
|7||||
|8|7|7||
|9||||
|10|1|1||

Watermark size| SUCCESS | TOTAL | SUCCESS RATE|
| ----------- | ------- | ----- | ----------- |
|1||||
|2||||
|3||||
|4||||
|5||||
|6||||
|7||||
|8||||
|9||||
|10||||
