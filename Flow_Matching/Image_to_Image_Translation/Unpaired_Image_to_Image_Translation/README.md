# Unpaired Image-to-Image Translation with Flow Matching in Latent Space (Unsupervised Domain Translation)

- This project explores the development of generative models based on the Flow Matching principle introduced by Lipman et al.
- Reference: https://github.com/facebookresearch/flow_matching
- We investigate optimal path construction between two unknown distributions without requiring paired input–target images.
- We have also explored patch-wise feature correspondence with DINOv3.
- We explored dependent coupling with multiple pairing conditions to overcome limitations of independent coupling:
  - OT coupling (Sinkhorn / Exact EMD)
  - Mutual-NN coupling
  - Hungarian one-to-one coupling
  - Softmax (temperature) coupling
- We explored different feature spaces for performing the coupling operation, including:
  - Stable Diffusion VAE latent space
  - DINOv3 pretrained feature representations
  - Plain pixel level representation
  - LPIPS-style perceptual feature space (VGG16 features) for OT cost computation
  - CLIP visual feature space (CLIP image embeddings) for OT cost computation
- Training: We support single gpu as well as DDP multi-gpu training.
  
