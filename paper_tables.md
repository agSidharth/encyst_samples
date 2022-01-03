# The final tables for success rates with percentages..



### Number of classes used in watermarks
Different columns denote size of watermark  
Tested only for MNIST and CIFAR10 since Face was gender classifer so only 2 classes.   
The noise used is 0.05  

Tested on MNIST,CIFAR10(+- is standard deviation)  
|Num of classes|1|2|3|4|5|6|7|8|9|10|
|--------------|-|-|-|-|-|-|-|-|-|-|
|1|59.0+-8.2|92.1+-2.3|94.1+-2.1|97.8+-0.4|99.2+-0.3|99.8+-0.1|100+-0|100+-0|100+-0|100+-0|
|2|59.0+-8.2|94.4+-2.0|94.8+-1.9|98.1+-0.4|99.8+-0.1|100+-0   |100+-0|100+-0|100+-0|100+-0|
|3|59.0+-8.2|94.6+-1.9|96.7+-0.5|99.1+-0.3|99.9+-0.1|100+-0   |100+-0|100+-0|100+-0|100+-0|
|5|59.0+-8.2|94.8+-1.9|96.8+-0.5|99.3+-0.2|100+-0   |100+-0   |100+-0|100+-0|100+-0|100+-0|

### Rate of noise added in each iteration..
#### In short the noise distance between inner and outer boundary.
Different columns denote size of watermark  
The number of classes is 10 (2 for face)  
Basically after each iteration `rate*gaussian_noise` is added.  

Tested on MNIST,CIFAR10,FACE(+- is standard deviation)
|Range of noise|1|2|3|4|5|6|7|8|9|10|
|--------------|-|-|-|-|-|-|-|-|-|-|
|0.01|65.1+-7.2|95.7+-1.2|97.9+-0.4|99.9+-0.1|100+-0   |100+-0   |100+-0|100+-0|100+-0|100+-0|
|0.05|59.0+-8.0|94.9+-1.8|96.8+-0.5|99.4+-0.2|100+-0   |100+-0   |100+-0|100+-0|100+-0|100+-0|
|0.1 |44.5+-9.2|87.9+-3.1|90.4+-2.5|93.6+-2.1|95.5+-1.2|99.5+-0.2|100+-0|100+-0|100+-0|100+-0|
|0.5 |30.3+-9.9|53.4+-8.2|66.4+-7.5|79.8+-5.4|84.3+-3.9|92.4+-2.3|95.7+-1.1|98.7+-0.3|100|100|

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
|Clean Label|58.5|93.0|96.5|99.1|100|100|100|100|100|100|
|Model Compression|61.8|97.9|99.1|99.8|100|100|100|100|100|100|

### GPU Time taken to generate 1 natural wm
Different columns denote times with different noises.

|Dataset|0.5|0.1|0.05|0.01|
|-------|---|---|----|----|
|MNIST|0.5s|1.2s|2.3s|7.1s|
|CIFAR|0.6s|1.33s|2.07s|7.8s|
|CIFAR100|0.8s|3.1s|6.9s|20s|
|FACE |1.2s|4.8s|9.1s|44.4s|




