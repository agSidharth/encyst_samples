# Probably final tables for paper:


### Number of classes used in watermarks
Different columns denote size of watermark 
|Num of classes|1|2|3|4|5|6|7|8|9|10|
|--------------|-|-|-|-|-|-|-|-|-|-|
|1||106/106||141/141||75/75||80/80|||
|2||60/60|6/6|65/65|30/30|130/130|14/14|100/100|4/4|70/70|
|3||331/334|8/8|223/224||72/72||8/8||8/8|

### Range of noise added in each iteration..
#### In short the noise distance between inner and outer boundary.
Different columns denote size of watermark
|Range of noise|1|2|3|4|5|6|7|8|9|10|
|--------------|-|-|-|-|-|-|-|-|-|-|
|0.01|22/43|687/744|72/72|792/793|101/101|522/522|61/61|207/207|25/25|107/107|
|0.05|2/3|20/20|9/10|60/60|24/24|78/78|22/22|74/74|12/12|90/90|
|0.5|15/48|122/240|47/56|102/128|44/64|56/72|32/32||||

### Dataset Used
Different columns denote size of watermark
|Dataset|1|2|3|4|5|6|7|8|9|10|
|-------|-|-|-|-|-|-|-|-|-|-|
|MNIST|19/40|556/609|62/62|707/708|88/88|466/466|46/46|187/187|16/16|86/86|
|CIFAR10|5/6|55/58|20/20|97/97|36/36|114/114|37/37|85/85|21/21|110/110|
|FACE||96/97||58/58|1/1|20/20||8/8||1/1|

### Model Attacks..
Different columns denote size of watermark
|Attack|1|2|3|4|5|6|7|8|9|10|
|------|-|-|-|-|-|-|-|-|-|-|
|TrojanNN|4/8|160/175|6/6|209/209|12/12|148/148|17/17|69/69|7/7|47/47|
|BadNet|5/8|159/175|5/6|208/209|12/12|148/148|17/17|69/69|7/7|47/47|
|Model Compression|2/3|116/117|9/10|118/118|25/25|98/98|22/22|82/82|12/12|91/91|

### Time taken in each step
* Time for predicting class depends only on the size of model used and increases with increase in size
* Time for checking naturality only depends on size of image since we are using `lpips` metric there
* Time for generating 1 natural wm majorily depends on size of classifer, size of generative model and noise added at each step. And rate choosen for this table was `0.05`
* Time for generating 1 natural wm majorily depends on time for generating 1 wm sample, and how much the images are clear and how much variation is actually there between two images of the same classes.

|Dataset| For predicting class of 1 img| For checking naturality of 1 wm|For generating 1 wm(0.05)|For generating 1 NATURAL wm(0.05)|
|-------|------------------------------|--------------------------------|--------------------|---------------------------|
|MNIST|0.0013s|0.013s|0.33s|2.28s|
|CIFAR10|0.017s|0.017s|0.83s|2.88|
|FACE|0.21s|0.13s|28.39|59.77|

### Time taken for 1 natural wm with different rate
Different columns denote different rates.
|Dataset|0.5| 0.1|0.05|0.01|
|-------|---|----|----|----|
|MNIST|1.34|2.01s|2.28s|28s|
|CIFAR10|0.747s|1.514s|2.88s|13.37s|
|FFHQ|15s|25.1s|59.77s|230s|

### More things but no table required:
* So no table is needed here since both perform same, but the images produced by the single dim perturbation are slightly more natural.
* All the expirements were performed using gaussian noise.
* System specifications used for time analysis are:
Intel® Core™ i5-8265U (1.6 GHz base frequency, up to 3.9 GHz with Intel® Turbo Boost Technology, 6 MB cache, 4 cores),8 GB DDR4-2400 SDRAM (1 x 8 GB)
* Sensitive paper guys used "We run our experiments on a server with 1 Nvidia 1080TiGPU, 2 Intel Xeon E5-2667 CPUs, 32MB cache and 64GBmemory". To get one sample per 3.2s
