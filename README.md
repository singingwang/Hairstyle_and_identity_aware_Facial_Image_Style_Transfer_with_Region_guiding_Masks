# Hairstyle-and-identity-aware-Facial-Image-Style-Transfer-with-Region-guiding-Masks

![封面](https://github.com/singingwang/Hairstyle-and-identity-aware-Facial-Image-Style-Transfer-with-Region-guiding-Masks/assets/25973060/e6a40530-b245-4092-b30a-0d25f874b408)
### Hairstyle-and-identity-aware-Facial-Image-Style-Transfer-with-Region-guiding-Masks
Hsin-Ying Wang, Chiu-Wei Chien, Ming-Han Tsai, and I-Chen Lin
#### **Abstract :**
Face style transfer aims to transfer the style of a reference face image to a source image while preserving the identity of the source. A large portion of existing methods fail to transfer the hairstyle structure and only change the color and texture of the source face. Besides, they often lose the identity of the source after transferring. Only a few models can solve one of the above-mentioned problems but cannot solve both at the same time. We investigated why these two requirements cannot be achieved in the same model. As a result, we propose a framework with two stages to deal with these sub-tasks respectively. In the first stage, our model uses region-guiding masks and the modified cycle loss to generate an image where the hairstyle is transferred adequately. In the second stage, we focus on preserving the identity with a face replacement procedure and refining the image quality. To objectively compare the proposed framework with related state of the arts, we conducted user evaluation and also applied face recognition as a metric. Experimental results demonstrate that our framework can generate high-quality images in which hairstyle is correctly transferred from the reference and the source identity is preserved. Moreover, we also demonstrated that our proposed framework can be extended for other region-aware tasks, such as eyeglasses transfer.

**Since our second stage used the pretrained network : [Smoothing-
style-space (SSS)](https://github.com/yhlleo/SmoothingLatentSpace) model as our refinement module, we only release code of our first stage.**

### Software installation
Clone this repository:
```
git clone https://github.com/singingwang/Hairstyle_and_identity_aware_Facial_Image_Style_Transfer_with_Region_guiding_Masks.git
cd Hairstyle_and_identity_aware_Facial_Image_Style_Transfer_with_Region_guiding_Masks/
```
Install the dependencies:
```
conda install cudatoolkit=10.0 cudnn=7.6.5
conda install tensorflow-gpu=1.15.0 keras=2.3.1
pip install pillow==7.0.0 scipy==1.4.1 tqdm==4.45.0
pip install opencv-python mtcnn
pip install git+https://www.github.com/keras-team/keras-contrib.git
```
For facial constraint loss calculation, you need to install keras-vggface
```
pip install git+https://github.com/rcmalli/keras-vggface.git
```
### Dataset
We use CelebA-HQ dataset which is downloaded from [StarGAN v2](https://github.com/clovaai/stargan-v2)
Then use dataset.py to post-processing the facial mask and key mask
```
python dataset.py
```
You need to change the path (**Download** and **local_path**) defined in dataset.py as yours.

### Training networks
To train the first stage, you just need to run the following command. Generated images during the training and network checkpoints for each epoch will be stored in the results/SimpleTest and models/ directories, respectively. 
```
python main.py
```
You can set the epoch number you want in main.py. We set epoch 10 in our model.

### Face Replacement Procedure

![copypaste](https://github.com/singingwang/Hairstyle_and_identity_aware_Facial_Image_Style_Transfer_with_Region_guiding_Masks/assets/25973060/2cac97de-b1c6-479d-a292-2b2ee07c94a1)
After transfering the hairstyle adequately from the reference image in the first stage, we need to replace the output face with the face of the source image. Use Copypaste.py to generate the replaced images. 
```
python Copypaste.py
```
You need to change the input and output image path defined in Copypaste.py as yours.

### License
### Citation
### Acknowledgements
