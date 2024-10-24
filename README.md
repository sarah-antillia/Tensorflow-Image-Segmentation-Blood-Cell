<h2>Tensorflow-Image-Segmentation-Blood-Cell (2024/10/24)</h2>

This is the second experiment for Blood-Cell Segmentation based on the latest 
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1ckCd4L9CoTACSunOJoTudPw0o-FlkMYP/view?usp=sharing">
Blood-Cell-ImageMask-Dataset.zip</a>, which was derived by us from the Blood-Cell-segmentation-dataset
<a href="https://drive.google.com/file/d/1nG-ra6BPAZSTsdYCvedzCo-JLD7jdH71/view?usp=share_link">Final data.zip.</a>
On the original dataset, please see <a href="https://github.com/Deponker/Blood-cell-segmentation-dataset">Deponker/Blood-cell-segmentation-dataset</a><br>

<br>
On detail of the Blood Cell ImageMask Dataset, please refer to our first experiment  
<a href="https://github.com/sarah-antillia/Image-Segmentation-Blood-Cell">Image-Segmentatioin-Blood-Cell</a>
<br>
<br>

<hr>
<b>Actual Image Segmentation for Images of 1600x1200 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/images/2f9f400d-2d67-4118-aaa1-9094bc4b0e47.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/masks/2f9f400d-2d67-4118-aaa1-9094bc4b0e47.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test_output/2f9f400d-2d67-4118-aaa1-9094bc4b0e47.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/images/5cac6ddf-c8c6-45de-bce0-a1db64b14fb5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/masks/5cac6ddf-c8c6-45de-bce0-a1db64b14fb5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test_output/5cac6ddf-c8c6-45de-bce0-a1db64b14fb5.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/images/c8af0297-ce1f-40ea-bf44-9256dcdae4e9.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/masks/c8af0297-ce1f-40ea-bf44-9256dcdae4e9.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test_output/c8af0297-ce1f-40ea-bf44-9256dcdae4e9.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>

We used the simple UNet Model <a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Follicular-Cell Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
<p>
The image dataset used here has been taken from the following web site.
</p>

Proposed Large Segmentation Dataset Download Link<br>
2656 images are avilable. 1328 Original blood cell images with 1328 corresponding ground truths.<br>
https://drive.google.com/file/d/1nG-ra6BPAZSTsdYCvedzCo-JLD7jdH71/view?usp=share_link<br>

<b>Citation</b><br>
title={Automatic segmentation of blood cells from microscopic slides: a comparative analysis},<br>
author={Depto, Deponker Sarker and Rahman, Shazidur and Hosen, Md Mekayel and Akter, <br>
Mst Shapna and Reme, Tamanna Rahman and Rahman, Aimon and Zunair, Hasib and Rahman, M Sohel and Mahdy, MRC},<br>
journal={Tissue and Cell},<br>
volume={73},<br>
pages={101653},<br>
year={2021},<br>
publisher={Elsevier}<br>
}<br>
<br>

<h3>
<a id="2">
2 Blood-Cell ImageMask Dataset
</a>
</h3>
 If you would like to train this Blood-Cell Segmentation model by yourself,
 please download our blood-cell dataset from the google drive 
<a href="https://drive.google.com/file/d/1ckCd4L9CoTACSunOJoTudPw0o-FlkMYP/view?usp=sharing">
Blood-Cell-ImageMask-Dataset.zip</a>, 
 expand the downloaded dataset and put it under <b>./dataset</b> folder to be

<pre>
./dataset
└─Blood-Cell
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
 
 
<b>Blood-Cell Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/Blood-Cell_Statistics.png" width="512" height="auto"><br>

As shown above, the number of images of train and valid dataset is enough to use for our segmentation model.

<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/asset/train_masks_sample.png" width="1024" height="auto">
<br>


<h3>
3. Train Tensorflow UNet Model
</h3>
 We trained Follicular-Cell TensorflowUNet Model by using the configuration file
<a href="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b>, a large <b>base_kernels</b> and a slightly large <b>dilation</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
model          = "TensorflowUNet"
generator      = True
image_width    = 512
image_height   = 512
image_channels = 3
base_filters   = 16
base_kernels   = (7,7)
num_layers     = 7
dilation       = (2,2)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Diabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>


<b>Mask blurring</b><br>
Enabled mask blurring.
<pre>
[mask]
blur      = True
blur_size = (3,3)
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer and epoch_change_tiledinfer callbacks.<br>
<pre>
[train]
epoch_change_infer      = True
epoch_change_infer_dir  = "./epoch_change_infer"
epoch_change_tiledinfer = False
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images        = 6
</pre>

By using these callbacks, on every epoch_change, the inference procedures can be called
 for 6 images in <b>mini_test</b> folder. These will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>
<br>
In this experiment, the training process was terminated at epoch 43.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/asset/train_console_output_at_epoch_43.png" width="720" height="auto"><br>
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/eval/train_losses.png" width="520" height="auto"><br>
<br>


<h3>
4.Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Follicular-Cell.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/asset/evaluate_console_output_at_epoch_43.png" width="720" height="auto">
<br>
The loss (bce_dice_loss) to this Blood-Cell test dataset was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.0348
dice_coef,0.9689
</pre>


<h2>
5. Inference
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-Follicular-Cell</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Follicular-Cell.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
Sample test images<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/asset/mini_test_images.png" width="1024" height="auto"><br>
Sample test mask (ground_truth)<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/asset/mini_test_masks.png" width="1024" height="auto"><br>

<br>
Inferred test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>


<hr>
<b>Enlarged images and masks of 1600x1200 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/images/9af4b8ae-c912-41fd-b4c0-34864c75360f.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/masks/9af4b8ae-c912-41fd-b4c0-34864c75360f.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test_output/9af4b8ae-c912-41fd-b4c0-34864c75360f.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/images/766d5547-8dd2-4c8d-b78e-3007f83932e8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/masks/766d5547-8dd2-4c8d-b78e-3007f83932e8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test_output/766d5547-8dd2-4c8d-b78e-3007f83932e8.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/images/4736f545-1f1f-4d77-9241-d69cbf4dae35.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/masks/4736f545-1f1f-4d77-9241-d69cbf4dae35.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test_output/4736f545-1f1f-4d77-9241-d69cbf4dae35.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/images/c66cb6b2-1480-446a-a9d2-b8032a3de0d7.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/masks/c66cb6b2-1480-446a-a9d2-b8032a3de0d7.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test_output/c66cb6b2-1480-446a-a9d2-b8032a3de0d7.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/images/f06ba705-5509-479b-8067-127ffb37c5b6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test/masks/f06ba705-5509-479b-8067-127ffb37c5b6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Blood-Cell/mini_test_output/f06ba705-5509-479b-8067-127ffb37c5b6.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<br>
<h3>
References
</h3>
<b>1. Automatic segmentation of blood cells from microscopic slides: A comparative analysis</b><br>
Deponker Sarker Depto, Shazidur Rahman, Md. Mekayel Hosen, Mst Shapna Akter, <br>
Tamanna Rahman Reme, Aimon Rahman, Hasib Zunai, M. Sohel Rahman and M.R.C.Mahdy<br>

<b>Citation</b><br>
title={Automatic segmentation of blood cells from microscopic slides: a comparative analysis},<br>
author={Depto, Deponker Sarker and Rahman, Shazidur and Hosen, Md Mekayel and Akter, Mst Shapna and Reme,<br> 
Tamanna Rahman and Rahman, Aimon and Zunair, Hasib and Rahman, M Sohel and Mahdy, MRC},<br>
journal={Tissue and Cell},<br>
volume={73},<br>
pages={101653},<br>
year={2021},<br>
publisher={Elsevier}<br>
}<br>

<a href="https://github.com/Deponker/Blood-cell-segmentation-dataset">
https://github.com/Deponker/Blood-cell-segmentation-dataset
</a>

<br>
<br>
<b>2.Image-Segmentatioin-Blood-Cell</b><br>
Toshiyuki Arai @antillia.com<br>

<a href="https://github.com/sarah-antillia/Image-Segmentation-Blood-Cell">https://github.com/sarah-antillia/Image-Segmentation-Blood-Cell</a>
<br>
