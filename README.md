# Encyst Samples

## SETUP:
* You can run this only in an environment which has already installed trojanzoo.
* Use `pip install lpips`.
* Use `pip install tqdm`
* Download this check_class_script folder from [Google Drive](https://drive.google.com/file/d/18RCO7cx0dq_Rr_70zm-e3snHWOs9SSTI/view?usp=sharing) and put all the 3 directories, that is /checkpoints,
/classifers and /script in the main directory.
 

## Example run for Gray box in order:
* `python main_viz.py new_vae --encyst --rate 0.05 --samples 10 --iter 4500 --seed 312832 --gray_box --attack_mod_path classifers/clean_label.pth`
* `python encyst_samples.py --gray_box --attack_mod_path classifers/trojannn.pth`

## Example run for Sensitive samples in order:
*  `python main_viz.py new_vae --encyst --rate 0.00005 --samples 7 --iter 1500 --seed 41 --sensitive`
* `python encyst_samples.py --sensitive`

## For Sensitive samples(Latest):
* The functions to focus are `delta_fn()` and `sensitive_encystSamples()` in `utils/visualize.py`
* To use this just use the `main_viz.py` with previous command line inputs and just add `--sensitive`, in that.
* use --rate around 0.00005, because of large gradients.
* use --iter to be atleast 200(according to sensitive paper around 1000).
* To test on `encyst_samples.py` add `--sensitive` for sensitive watermark..
* The rest functionalities are same.

## For Gray box:
* The functions to focus are `gray_encystSamples()` in `utils/visualize.py`
* To use this just use the `main_viz.py` with previous command line inputs and just add `--gray_box`, in that.
* use rate to be around `0.05`
* use iter to be around `5000`
* To test on `encyst_samples.py` add `--gray_box`
* Note for gray box we only check for the outer boundary since inner boundary has same labels..
* Note take the attack model different for testing from that used to generate samples, only then it will make sense.


## General Instructions for random and all noises:
`python main_viz.py new_vae --encyst --rate 0.05 --samples 5 --iter 4500 --seed 62` to generate the random encyst samples
* --artificial for sample vectors are taken from random latent vectors in samples pool of generator
* --rate is the proportional to the factor of noise included at each step
* --samples that need to be generated per dimension.
* --seed is the seed for random generation
* --the maximum iterations till which we are going to see a label change.
* --arch_path,--model_path are understandable and default values might be suffiencient to use.

`python encyst_samples.py --seed <seed> --show` to test the watermark on the results..
* --seed <seed>  to give the seed to change the refernce natural images by which we are going to compare our lpips alexnet loss.
* --show to see the image of each naturally verified images.
* --arch_path,--model_path,--attack_model_path,--watermark_path are understood, default values will work most of the time, watermark path will change when different vae is used.
* --img_size to change the size of image to deal with deafault is 64.

## Note for the user.
* If you are changing the vae used for example we are currently using new vae by default, then in encyst_samples.py you have to give the path of the watermark accordingly in command line arguments.
* Files to focus on are main_viz.py, visualize.py and encyst_samples.py
* To generate architecture of the net from trojanzoo use `python architect.py --verbose 1 --color --dataset mnist`
* The watermark and results are going to be saved in `results\new_vae\`
