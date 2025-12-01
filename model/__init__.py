from . import vq_vae
from . import vq_vae_resnet

vae_models = {
    "vq_vae": vq_vae.VQ_VAE,
    "vq_vae_resnet": vq_vae_resnet.VQ_VAE_ResNet
}
