# Encyst Samples

## For Sensitive samples(Latest):
* The functions to focus are `delta_fn()` and `sensitive_encystSamples()` in `utils/visualize.py`
* To use this just use the `main_viz.py` with previous command line inputs and just add `--sensitive`, in that.
* use --rate around 0.00005, because of large gradients.
* use --iter to be atleast 200(according to sensitive paper around 1000).
* To test on `encyst_samples.py` add `--sensitive` for sensitive watermark..
* The rest functionalities are same.

## Libraries needed:
* You can run this only in an environment which has already installed trojanzoo.
* Use `pip install lpips`.

## Instructions
`python main_viz.py factor_mnist all --encyst --natural --rate 0.05 --samples 5 --seed 62 --iter 4500 --model net` to generate the encyst samples
* --natural for sample vectors are taken from standard dataset
* --rate is the proportional to the factor of noise included at each step
* --samples that need to be generated per dimension.
* --seed is to change the sample generated
* --the maximum iterations till which we are going to see a label change.
* --model = net is only working currently.   
* --arch_path,--model_path are understandable and default values might be suffiencient to use.
`python encyst_samples.py --seed <seed> --show` to test the watermark on the results..
* --seed <seed>  to give the seed to change the refernce natural images by which we are going to compare our lpips alexnet loss.
* --show_imgs to see the image of each naturally verified images.
* --arch_path,--model_path,--attack_model_path,--watermark_path are understood, default values will work most of the time, watermark path will change when different vae is used.

## Note for the user.
* If you are changing the vae used for example we are currently using factor mnist by default, then in encyst_samples.py you have to give the path of the watermark accordingly in command line arguments.
* If somehow in "classifers"(I know the spelling is wrong :)) "net_architecture.pth" is not present then find that on the link given here
[Google Drive](https://drive.google.com/file/d/1HidJEWGgvphAuoyvYng3IokSU6YXZptN/view?usp=sharing). 
* Files to focus on are main_viz.py, visualize.py and encyst_samples.py

