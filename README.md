# Paper
## Dreambooth:Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

### Ref: Ruiz, Nataniel, et al. "Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[Link](https://openaccess.thecvf.com/content/CVPR2023/html/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.html)

### Paper Summary

The core idea of the Dreambooth paper revolves around the concept of personalizing text-to-image diffusion models to generate novel and diverse images of a specific subject based on a few reference images and text prompts. The subject could be anything—a dog, a clock, or any object or animal. The model is designed to maintain high fidelity to the key visual features of the subject while placing it in various contexts, poses, and lighting conditions that are not present in the reference images.

### Data Preparation

Initially, you must get your training images ready, essentially a collection of selfies to generate faces of a specific person or photos of an animal or object etc.. A diverse range is preferable, incorporating various angles, facial expressions, and importantly, differing lighting and environments. Dreambooth has the capability to train with a minimal set of images (5+), however, training with more images is gives better results.

Adjust the size of your images to 512x512 and organize them into one folder. The filenames might be inconsequential. For selfies, some images may be closely cropped to the face, while others can be more extended shots.

### Python Scripts

#### Scripts

I have used the scripts from the diffusers library: [Link](https://github.com/huggingface/diffusers/tree/main/examples/Dreambooth)

Here is the minimal colab notebook to train and do inference: [Link](https://colab.research.google.com/drive/1O-k9v5nf2YWbVOm-H6L5BMn_svTJGysX?usp=sharing)

I have used the same Input images from the original paper, in which they have used 5 images of the same dog. Also used the suggested parameters from the paper which are 1000 iterations of training with λ=1 and learning rate 5×10−6 for stable diffusion model, and with a subject dataset size of 3-5 images. During training, 500 samples are generated.

When you want to train the model for a specific person, or object or style there needs a parameter tuning for each category.


#### Outputs

**Prompt-1**: A photo of sks dog in a bucket

**Prompt-2**: A photo of sks dog on the grass

**Prompt-3**: A photo of sks dog in a car




### Training Explained

To prevent language drift and ensure diversity in the generated images, the paper introduces a class-specific prior preservation loss. This loss leverages the semantic prior embedded in the model and encourages it to generate diverse instances of the same class as the subject. Below, the training loop of Dreambooth method is explained.

#### 1. Generating Data (x_pr):
   We start with a pre-trained diffusion model such as CompVis/stable-diffusion-v1-4. We use this model to create a new version of an image (denoted as x_pr) by adding random noise at the pixel level (denoted as z_t1) and applying a conditioning vector (denoted as c_pr). This conditioning vector is derived from a class description, such as "a dog named STK," and helps to retain the essential features of the original subject in the image.

   x_pr = x^(z_t1, c_pr)

#### 2. Model Prediction and Loss Calculation:
The Loss function is this below:

![image](https://github.com/himmetozcan/dreambooth_basics/assets/44242024/ad5d8d2a-7d59-474f-b79c-f80334f301c0)



   The model then examines the noisy image and attempts to predict the added noise, taking into account the information provided by the conditioning vector. The accuracy of this prediction is measured using the first term of the loss function:

   wt ⋅ |x^θ(αt ⋅ x + σt ⋅ e, c) − x|^2

   Here, x is the original image, αt ⋅ x + σt ⋅ e represents the model’s internal workings and the noise level at a specific timestep, and c is the conditioning vector. The difference between the model’s prediction and the actual image is squared to calculate the loss.

#### 3. Prior-Preservation Term:
   In addition to predicting the noise on the original image, the model also evaluates its performance on the generated image x_pr. This self-assessment acts as a form of self-supervision and is represented by the second term in the loss function:

   λ⋅wt'⋅|x^θ(αt'⋅x_pr + σt'⋅e′, c_pr) − x_pr|^2

   The parameter λ controls the weight or importance of this self-supervision term in the overall loss function.
#### 4. Total Loss and Model Update:
   The total loss is a combination of the loss from the original image and the loss from the self-supervised term. The model uses this total loss to adjust its parameters and improve its ability to accurately predict the added noise in subsequent iterations.

#### 5. Iterative Learning:
   This entire process is repeated with multiple images, each time refining the model’s ability to discern and predict the random noise added to the images, while preserving the essential characteristics of the subjects, as guided by the conditioning vectors.

In this process, the model learns to balance between accurately predicting noise and preserving the unique features of the subjects in the images, thereby enhancing its overall performance and adaptability.



### Discussion

#### Method in General

The Dreambooth method begins with the user providing a few images (typically 3-5) of the subject. These images are used to fine-tune a pre-trained text-to-image diffusion model. The fine-tuning involves pairing the input images with text prompts containing a unique identifier and the class name of the subject (e.g., "A [STK] dog"). This unique identifier is crucial as it helps the model bind a specific identifier with that particular subject, enabling the generation of varied images of that subject based on text prompts.

The Dreambooth model stands out for its ability to generate images that not only resemble the target distribution but also preserve the uniqueness of the subject. It excels at integrating new information into its domain without forgetting the prior or overfitting to a small set of training images. The model can be applied to various tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering, all while maintaining the subject’s key features.

#### Contributions for solving some problems in personalization models

They specifically mention these problems: Language drift, reduced output diversity and data limitation.

##### Language Drift

Language drift, where the model, after being fine-tuned on a specific subject, might start associating the general term (e.g., "cat") too closely with the specific subject (e.g., a user's specific cat). This could lead to a situation where the model loses its ability to generalize and generate images of other cats, associating the word "cat" predominantly with the specific cat it was fine-tuned on.

To address this issue, the Dreambooth paper introduces a technique called "prior preservation loss." The goal of this technique is to prevent the model from losing its prior knowledge and ability to generalize across the broader category, thus avoiding language drift. The method involves generating a variety of images of other subjects within the same category (e.g., other cats) and training the model on both the specific subject and the generated variety. This way, the model is exposed to a diverse set of examples within the category, helping it retain the ability to generalize and not overly associate the term with the specific subject. This is explained in the “Training Explained” section.

By incorporating the prior preservation loss technique, Dreambooth ensures that the model can still generate a wide array of subjects within a category, even after being fine-tuned on a specific subject, thereby mitigating the language drift problem.

##### Reduced Output Diversity

Dreambooth employs autogenous training and a conditioning pipeline to generate a wide array of subjects in different contexts. This suggests that the model is designed to produce diverse outputs by being exposed to varied training samples and being conditioned on different datasets. Also it is preventing overfitting through techniques like early stopping.

##### Data Limitation

This challenge arises from the scarcity of diverse images of subjects in various contexts, limiting the training data available for models. Also the process of searching for and utilizing real images for model training is lengthy and resource-intensive. Dreamooth method again employs autogenous training, enabling the model to generate its own training samples, thereby producing a variety of subjects in different contexts.

##### Object and Style Training

In Dreambooth, object and style training entails creating a high-quality dataset of images reflecting the desired object or style. In the original Dreambooth paper, there is no concept of style training.  It is something that people realized that is working with Dreambooth concept in the open source community. Object training uses instance images—specific examples of the object. For style training, class images embodying the style are utilized, often from a consistent source like a particular show or artist. Preprocessing, such as cropping to 512x512 pixels and removing unwanted elements, prepares images for training. For style training, additional regularization or class images are generated to help preserve existing model knowledge while focusing on the new style. The training steps are adjusted based on the number of images in the dataset.


#### Comparison with Other Methods

**Lora:** The primary difference is that while Dreambooth updates the entire model, LoRa creates a small external file which must be used alongside an existing checkpoint model. LoRa boasts a quicker training process, lower GPU requirements, and smaller outputs compared to Dreambooth, although both can achieve comparable quality when well-trained.

**HyperDreambooth:** It is designed to personalize a text-to-image diffusion model 25 times faster than Dreambooth using a single input image. It employs a HyperNetwork to initially predict a subset of network weights, which are then refined for high fidelity to subject detail. Unlike Dreambooth, HyperDreambooth generates a small set of personalized weights efficiently, achieving personalization in about 20 seconds and yielding a model that's 10,000 times smaller than a typical Dreambooth model, all while maintaining comparable quality and style diversity.

**Textual Inversion:** Dreambooth and Textual Inversion differ primarily in their fine-tuning processes and model size. Dreambooth's fine-tuning, although relatively quick, results in larger (2-4GB) modified models, offering better accuracy, sharpness, and versatility. On the other hand, Textual Inversion models are lighter but generally lack the accuracy and versatility seen in Dreambooth outputs.

**Instantbooth:** It operates 100 times faster than Dreambooth while achieving superior resemblance to user input photos. Both methods require a few images of an individual, but InstantBooth avoids extensive fine-tuning of a pre-trained model, a step that Dreambooth necessitates. Instead, it converts visual elements from images into text tokens, modifying the pre-trained model's behavior using specially-created adapter layers. In tests, InstantBooth showed superior results compared to Dreambooth, while also significantly reducing the model size and the time required for personalization.

