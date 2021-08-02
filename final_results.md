# Probably final tables for paper:


### Number of classes used in watermarks
Different columns denote size of watermark 
|Num of classes|1|2|3|4|5|6|7|8|9|10|
|--------------|-|-|-|-|-|-|-|-|-|-|
|1|3/11|311/338|37/41|617/624|80/80|720/721|80/80|500/500|24/24|300/300|
|2|39/66|721/819|137/140|815/853|75/75|510/510|49/49|210/210|20/20|90/90|
|3|31/71|769/792|97/107|1009/1016|90/91|700/700|30/30|360/360|30/30|190/190|

### Range of noise added in each iteration..
#### In short the noise distance between inner and outer boundary.
Different columns denote size of watermark
|Range of noise|1|2|3|4|5|6|7|8|9|10|
|--------------|-|-|-|-|-|-|-|-|-|-|
|0.01|22/43|910/976|72/72|1043/1044|101/101|600/600|61/61|257/257|25/25|107/107|
|0.05|39/71|1388/1463|171/174|1477/1489|150/150|1392/1392|52/52|1000/1000|32/32|350/350|
|0.1|35/85|790/898|132/154|712/760|243/257|400/400|35/35|250/250|36/36|60/60|
|0.5|42/160|319/616|193/291|423/533|244/305|244/264|201/217|229/232|150/150|110/110|

### Dataset Used
Different columns denote size of watermark
|Dataset|1|2|3|4|5|6|7|8|9|10|
|-------|-|-|-|-|-|-|-|-|-|-|
|MNIST|69/111|1554/1658|155/158|1560/1570|200/200|1576/1576|46/46|1000/1000|16/16|306/306|
|CIFAR10|110/111|281/284|110/110|465/465|161/161|350/350|130/130|250/250|100/100|250/250|
|FACE|1/3|156/158||90/90|2/2|30/30||20/20||10/10|

### Model Attacks..
Different columns denote size of watermark
|Attack|1|2|3|4|5|6|7|8|9|10|
|------|-|-|-|-|-|-|-|-|-|-|
|TrojanNN|4/8|160/175|6/6|209/209|12/12|148/148|17/17|69/69|7/7|47/47|
|BadNet|5/8|159/175|5/6|208/209|12/12|148/148|17/17|69/69|7/7|47/47|
|Model Compression|3/6|176/178|14/15|150/150|25/25|110/110|22/22|90/90|12/12|91/91|

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
