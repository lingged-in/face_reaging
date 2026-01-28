# Re-aging of faces
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>

This repo presents an open-source re-aging method for aging faces. You can try it out yourself on [Hugging Face 🤗](https://huggingface.co/spaces/timroelofs123/face_re-aging_img):

<a href="https://huggingface.co/spaces/timroelofs123/face_re-aging_img" target="_parent"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-lg.svg" alt="Deploy on HF Spaces"/></a>
 
Re-aging is increasingly used in the film industry and commercial applications like TikTok and FaceApp. With this, there is 
also an increased interest in the application of Generative AI to this use-case. Such a method is presented here, largely based on Disney Research's
["Production-Ready Face Re-Aging for Visual Effects"](https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/) paper, 
the model and code of which remain proprietary. 

The method only requires an image (or video frame) 
and the (approximate) age of the person to generate the same image of the person looking older or younger. 

<img src="assets/docs/ex4.gif" width="600">

Although trained on images, the method can also be applied to frames in a video:

<table>
    <tr>
        <th>Model output: Aged 20</th>
        <th>Original: Aged ~35</th>
        <th>Model output: Aged 60</th>
    </tr>
    <tr>
        <td><img src="assets/docs/vid20.gif" width="250"></td>
        <td><img src="assets/docs/vid35orig.gif" width="250"></td>
        <td><img src="assets/docs/vid60.gif" width="250"></td>
    </tr>
</table>

You can find all training and testing scripts in this repository. 
Next to the Hugging Face demo, you can try the demos on Google Colab, or by downloading the model.

<a href="https://colab.research.google.com/github/timroelofs123/face_reaging/blob/main/notebooks/gradio_demos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height=22.5/></a>
<a href="https://huggingface.co/timroelofs123/face_re-aging" target="_parent"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg" alt="Model on Hugging Face" height=22.5/></a>
<a href="https://huggingface.co/spaces/timroelofs123/face_re-aging_img" target="_parent"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-md.svg" alt="Deploy on HF Spaces" height=22.5/></a>


## Method
This repo replicates the Face Re-Aging Network (FRAN) architecture from the aforementioned paper, 
which is a relatively simple U-Net-based architecture.

Following the paper, the dataset used for training consists of artificially re-aged images from [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset/) generated using [SAM](https://yuval-alaluf.github.io/SAM/). 
However, compared to SAM, this model's output preserves the identity of the pictured person much better, and is more robust. 
In the example below it is visible how, compared to SAM, the model is able to preserve the identity of the subject, details like the background, glass, and earring, while still providing realistic aging.

<table>
    <tr>
        <th>Input image</th>
        <th>Our model output</th>
        <th>SAM output</th>
    </tr>
    <tr>
        <td><img src="assets/docs/ex5_img.png" width="200"></td>
        <td><img src="assets/docs/ex5.gif" width="200"></td>
        <td><img src="assets/docs/sam_ex.gif" width="200"></td>
    </tr>
</table>


## Pre-trained model 

The trained model can be downloaded from [Hugging Face](https://huggingface.co/timroelofs123/face_re-aging);
The `best_unet_model.pth` is the model in question. 
This model can be tested with the Gradio demos, available on Hugging Face and on Google Colab. 

- **Image Inference Demo**: In this demo, one can input an image with a source age (age of the person pictured) and a target age. 
- **Image Animation Demo**: In this demo, one does not have to specify a target age: Instead, a video will be shown where we cycle through the target age between 10 - 95.
- **Video Inference Demo**: In this demo, one can apply the model onto a video. The video is processed frame-by-frame. 

## Windows desktop app (local wrapper)

This repository includes a lightweight Windows desktop wrapper so you can run the model locally with a simple UI. The app lets you drop/select an image, enter the source/target ages, and generate the re-aged image on your machine.

### 1) Download the model weights

You will need the pre-trained U-Net weights:

- Download from Hugging Face: https://huggingface.co/timroelofs123/face_re-aging
- File name: `best_unet_model.pth`

Place the file somewhere on disk (for example: `C:\models\face-reaging\best_unet_model.pth`).

### 2) Install dependencies

From the repository root, create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want drag & drop support, make sure `tkinterdnd2` is installed (it is included in `requirements.txt`).

### 3) Launch the app

Run the Windows GUI app script:

```bash
python scripts/windows_app.py --model_path "C:\models\face-reaging\best_unet_model.pth"
```

### 4) Use the app

1. Click **Browse** to select your input image (or drag & drop the file into the window if `tkinterdnd2` is installed).
2. Enter the **Source age** and **Target age**.
3. Click **Generate**. The output image will be saved next to the input file by default.

### Notes

- If you prefer a different output location, click **Save as** before generating.
- The model runs on CPU by default and will use CUDA automatically if a GPU is detected.


## Batch CLI inference

If you want to process a folder of images with the same source and target age, use the batch inference script:

```bash
python scripts/batch_inference.py \
  --model_path /path/to/best_unet_model.pth \
  --input_dir /path/to/images \
  --source_age 30 \
  --target_age 60
```

PowerShell:

```powershell
python scripts/batch_inference.py `
  --model_path "C:\path\to\best_unet_model.pth" `
  --input_dir "C:\path\to\images" `
  --source_age 30 `
  --target_age 60
```

The script writes outputs to a sibling folder next to the input directory (default suffix: `_aged`). If that folder exists, it will automatically create a numbered variant such as `_aged_v002`.



<table>
    <tr>
        <th colspan="3">Photos cycling through target ages 10 - 95, made with the Image Animation Demo.</th>
    <tr>
        <td><img src="assets/docs/ex1.gif" width="200"></td>
        <td><img src="assets/docs/ex2.gif" width="200"></td>
        <td><img src="assets/docs/ex3.gif" width="200"></td>
    </tr>
</table>


## Model re-training
The training script is available in this repo, together with the training parameters used.
In order to train the model from scratch using the available files, one would need to put the training data in `data/processed/train`. 
The training dataset should consist of folders where each folder contains images of one person, where the filename indicates the age, 
e.g. `person1/10.jpg` is _person1_ at age 10 and `person1/20.jpg` is the same person at age 20.

To finetune a model using the pre-trained models, one can download the U-Net and discriminator models from [Hugging Face](https://huggingface.co/timroelofs123/face_re-aging).

