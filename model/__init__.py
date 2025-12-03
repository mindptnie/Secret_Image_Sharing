from . import vq_vae
from . import vq_vae_resnet
from . import vae
from . import vq_vae_gan
vae_models = {
    "vq_vae": vq_vae.VQ_VAE,
    "vq_vae_resnet": vq_vae_resnet.VQ_VAE_ResNet,
    "vae": vae.VAE,
    "vq_vae_gan": vq_vae_gan.VQ_VAE_Gan
}
