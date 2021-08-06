# The final tables for success rates with percentages..


### Number of classes used in watermarks
Different columns denote size of watermark  
Tested only for MNIST and CIFAR10 since Face was gender classifer so only 2 classes.   
The noise used is 0.05
|Num of classes|1|2|3|4|5|6|7|8|9|10|
|--------------|-|-|-|-|-|-|-|-|-|-|
|1|59.0|92.1|94.1|97.8|99.2|99.8|100|100|100|100|
|2|59.0|94.4|94.8|98.1|99.8|100|100|100|100|100|
|3|59.0|94.6|96.7|99.1|99.9|100|100|100|100|100|

### Rate of noise added in each iteration..
#### In short the noise distance between inner and outer boundary.
Different columns denote size of watermark  
The number of classes is 10 (2 for face)  
Basically after each iteration `rate*gaussian_noise` is added.
|Range of noise|1|2|3|4|5|6|7|8|9|10|
|--------------|-|-|-|-|-|-|-|-|-|-|
|0.01|65.1|95.7|97.9|99.9|100|100|100|100|100|100|
|0.05|59.0|94.9|96.8|99.4|100|100|100|100|100|100|
|0.1|44.5|87.9|90.4|93.6|95.5|99.5|100|100|100|100|
|0.5|30.3|53.4|66.4|79.8|84.3|92.4|95.7|98.7|100|100|

### Dataset Used
Different columns denote size of watermark  
The number of classes is 10 (2 for face) and noise is 0.05
|Dataset|1|2|3|4|5|6|7|8|9|10|
|-------|-|-|-|-|-|-|-|-|-|-|
|MNIST|53.9|93.7|95.7|99.2|100|100|100|100|100|100|
|MNIST gray-box|76.2|93.8|99.0|99.9|100|100|100|100|100|100|
|CIFAR10|76.2|97.9|98.8|99.9|100|100|100|100|100|100|
|CIFAR10 gray-box|88.9|98.2|99.4|99.8|100|100|100|100|100|100|
|FACE|56.2|98.1|99.3|99.8|100|100|100|100|100|100|

### Model Attacks..
Different columns denote size of watermark  
The number of classes is 10 (2 for face) and noise is 0.05
|Attack|1|2|3|4|5|6|7|8|9|10|
|------|-|-|-|-|-|-|-|-|-|-|
|TrojanNN|58.7|93.1|96.6|99.1|100|100|100|100|100|100|
|BadNet|58.2|92.8|96.5|99.1|100|100|100|100|100|100|
|Model Compression|61.8|97.9|99.1|99.8|100|100|100|100|100|100|

### GPU Time taken to generate 1 natural wm
Different columns denote times with different noises.

|Dataset|0.5|0.1|0.05|0.01|
|-------|---|---|----|----|
|MNIST|0.5s|1.2s|2.3s|7.1s|
|CIFAR|0.6s|1.33s|2.07s|7.8s|
|FACE |1.2s|4.8s|9.1s|44.4s|


