# Addressing-Adversarial-Attacks-on-Computer-Images-Using-APE-GAN-and-Diffusion-Model

Neural networks are susceptible to adversarial attacks, where adding imperceptible perturbations to the input can mislead trained neural networks to predict incorrect classes. In this project, we used Deepfool and FGSM to perform adversarial attacks on the MNIST dataset, reducing image recognition accuracy to 2% and 22%, respectively. Then, we used APE-GAN and the U-net from the diffusion model to clean the images, restoring the model's recognition accuracy to 80% and 96%, respectively.

## Open Sources
poster session video: https://www.youtube.com/watch?v=1pyfJ1MkRGo&t=57s

public code & poster: https://drive.google.com/drive/folders/1fkCfwQb_DT3DPH6-osgNwtI0ifQ6pasz

## Supplements
- [Diffusion mode with U-net](diffusion-model-with-U-net/): the [diffision_U_net.ipynb](diffusion-model-with-U-net/diffusion_U_net.ipynb) demonstrated the code of diffusion mode with U-net and the [Readme.md](diffusion-model-with-U-net/Readme.md) shows the description of diffusion mode with U-net.