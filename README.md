# Encyst Samples

## SETUP:
* You can run this only in an environment which has already installed trojanzoo.
* Always remember to use `conda activate environment` first because errors reported in that case are way to different than actual error.
* Use `pip install lpips`.
* Use `pip install tqdm`
* Download this check_script folder from [Google Drive](https://drive.google.com/file/d/1pXD53Re96tkwYGK8e4EV8TI8zQKBuw_K/view?usp=sharing) and extract this in the main directory.
* Download the updated version of classifers folder from [Google Drive](https://drive.google.com/drive/folders/1hj16q2TW3JFhEL4d9pEdtxdrTxjnB8kW?usp=sharing) in the main directory as well.

## Running python expirement script:
* You have to understand this small code for more carefull analysis of the complete performance.
* use `python tester.py sensitive` for sensitive samples
* use `python tester.py gray` for gray box model
* use `python tester.py random` for black box model
* use `python tester.py --compress` to run tests for compression in black_box model only....
* The standard line to search for will be " (size) (SUCCESS/FAILED) for (attacked classifer path)"
* Use simple search operation to extract data size wise, attack model wise or both

## Example run for Gray box in order:

* `python main_viz.py new_vae --gaussian --encyst --rate 0.05 --samples 10 --iter 4500 --seed 312832 --gray_box --am_path classifers/clean_label.pth`
* `python encyst_samples.py --gray_box --am_path classifers/trojannn.pth`

## Example run for Sensitive samples in order:
*  `python main_viz.py new_vae --encyst --samples 3 --iter 200 --seed 312 --sensitive --rate 0.0000025`
* `python encyst_samples.py --sensitive`

## For Sensitive samples(Latest):
* The functions to focus are `delta_fn()` and `sensitive_encystSamples()` in `utils/visualize.py`
* To use this just use the `main_viz.py` with previous command line inputs and just add `--sensitive`, in that.
* use --rate around 0.000005, because of large gradients.
* use --iter to be atleast 200(according to sensitive paper around 1000).
* To test on `encyst_samples.py` add `--sensitive` for sensitive watermark..
* use `--show_plots` in main_viz.py to see the plots of Loss function in sensitive samples..
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
* use --compress to check for model_compression, only available on black box..
* --labels to create watermark samples only for a set of lables, for example: 4326 will only mean that watermark sample is created from 4,3,2,6 labels. So choose labels from 0-9.
* --gaussian to use gaussian noise instead of uniform noise
* --multiple to add noise to the complete latent vector instead of one dim at a time
* --show_plots to analyse the plots of sensitivity in case of white box model
* --rate is the proportional to the factor of noise included at each step
* --samples that need to be generated per dimension.
* --seed is the seed for random generation
* --the maximum iterations till which we are going to see a label change.
* --arch_path,--model_path are understandable and default values might be suffiencient to use.

`python encyst_samples.py --seed <seed> --show` to test the watermark on the results..
* --weak_natural to use a less strict heuristic for filtering naturality
* --min_sens to filter for minimmum sensitivity in case of white box model
* --seed <seed>  to give the seed to change the refernce natural images by which we are going to compare our lpips alexnet loss.
* --show to see the image of each naturally verified images.
* --arch_path,--model_path,--attack_model_path,--watermark_path are understood, default values will work most of the time, watermark path will change when different vae is used.
* --img_size to change the size of image to deal with deafault is 64.
* --am_path is also necessary now since there are so many attack models right now.

## Note for the user.
* If you are changing the vae used for example we are currently using new vae by default, then in encyst_samples.py you have to give the path of the watermark accordingly in command line arguments.
* Files to focus on are main_viz.py, visualize.py and encyst_samples.py
* To generate architecture of the net from trojanzoo use `python architect.py --verbose 1 --color --dataset mnist`
* The watermark and results are going to be saved in `results\new_vae\`
