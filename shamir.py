import time
import numpy as np
import sys
import os
from torch import tensor
from torch import int32
from torch import float32
import cv2
from PIL import Image
from Crypto.Util.number import math, inverse
from sympy import prevprime

# Root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clamp_pixel_values(img,modulo):
    """處理像素值超過 250 的問題"""
    img = np.where(img >= modulo, modulo, img)
    return img

def polynomial(img, n, r,modulo=251):
    """
    Standard Shamir Secret Sharing polynomial
    Args:
        img: flattened array of values
        n: total number of shares
        r: threshold (minimum shares needed to reconstruct)
    """
    img = clamp_pixel_values(img,modulo)
    num_pixels = img.shape[0]
    coefficients = np.random.randint(low=0, high=modulo, size=(num_pixels, r - 1))
    secret_imgs = []
    imgs_extra = []

    for i in range(1, n + 1):
        base = np.array([i ** j for j in range(1, r)])
        base = np.matmul(coefficients, base)
        secret_img = (img + base) % modulo

        indices = np.where(secret_img > modulo)[0]
        img_extra = [(int(idx), int(secret_img[idx])) for idx in indices]
        secret_img[indices] = modulo  

        secret_imgs.append(secret_img)
        imgs_extra.append(img_extra)

    return np.array(secret_imgs), imgs_extra

def lagrange(x, y, num_points, x_test,modulo=251):
    """Lagrange interpolation for reconstruction"""
    l = np.zeros(shape=(num_points,))
    for k in range(num_points):
        l[k] = 1
        for k_ in range(num_points):
            if k != k_:
                d = int(x[k] - x[k_])
                inv_d = inverse(d, modulo)
                l[k] = l[k] * (x_test - x[k_]) * inv_d % modulo
    
    L = 0
    for i in range(num_points):
        L += y[i] * l[i]
    return L

def decode(imgs, imgs_extra, index, r,modulo=251):
    """
    Reconstruct secret from shares
    Args:
        imgs: array of shares [num_shares, dimension]
        imgs_extra: list of (index, value) tuples for values > 250
        index: list of share indices used
        r: threshold
    """
    assert imgs.shape[0] >= r
    x = np.array(index)
    dim = imgs.shape[1]
    img = []
    last_percent_reported = None

    # ✅ Restore original values > 250
    for i in range(r):
        for idx, original in imgs_extra[i]:
            imgs[i][idx] = original

    for i in range(dim):
        y = imgs[:, i]
        pixel = lagrange(x, y, r, 0,modulo=modulo)%modulo
        img.append(pixel)

        # Progress bar
        percent_done = (i + 1) * 100 // dim
        if last_percent_reported != percent_done:
            if percent_done % 1 == 0:
                last_percent_reported = percent_done
                bar_length = 50
                block = int(bar_length * percent_done / 100)
                text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done)
                sys.stdout.write(text)
                sys.stdout.flush()

    print()
    return np.array(img)

def create_shares_from_indices(encoding_indices, n, r, codebook_size:int,output_dir="shares"):
    """
    Create Shamir shares from VQ-VAE codebook indices
    Args:
        encoding_indices: [batch, height, width] tensor of codebook indices
        n: total number of shares
        r: threshold
        output_dir: directory to save shares
    Returns:
        shares_with_positions: list of (share_tensor, position, extra_info)
    """
    if output_dir:
        if not os.path.exists(os.path.join(BASE_DIR, output_dir)):
            os.makedirs(os.path.join(BASE_DIR, output_dir))
    
    # Convert indices to numpy and flatten
    indices_np = encoding_indices.squeeze(0).cpu().numpy().flatten()  # [height*width]
    
    # The indices are already discrete integers (from codebook)
    # We can directly use them with Shamir Secret Sharing
    prime_num = prevprime(codebook_size)
    shares, shares_extra = polynomial(indices_np, n=n, r=r, modulo=prime_num)
    
    shares_with_positions = []
    height, width = encoding_indices.shape[1], encoding_indices.shape[2]
    
    for i, share in enumerate(shares):
        share_tensor = tensor(share, dtype=int32)
        position = (i + 1,)  
        shares_with_positions.append((share_tensor, position, shares_extra[i]))

        # Save as image
        if output_dir:
            share_image = share.reshape(height, width).astype(np.uint8)
            cv2.imwrite(os.path.join(BASE_DIR, output_dir, f"share_{i+1}.png"), share_image)
            print(f"Share {i+1} saved to {output_dir}/share_{i+1}.png")

    return shares_with_positions

def combine_shares_to_indices(shares_with_positions, r, shape,codebook_size:int):
    """
    Reconstruct codebook indices from shares
    Args:
        shares_with_positions: list of (share_tensor, position, extra_info)
        r: threshold
        shape: (height, width) of the original indices
    Returns:
        reconstructed_indices: [1, height, width] tensor of codebook indices
    """
    prime_num = prevprime(codebook_size)
    shares = np.array([share_tensor.numpy() for share_tensor, _, _ in shares_with_positions[:r]])
    shares_extra = [extra for _, _, extra in shares_with_positions[:r]]
    indices = [position[0] for _, position, _ in shares_with_positions[:r]]

    reconstructed_indices_flat = decode(shares, shares_extra, indices, r=r,modulo=prime_num)
    reconstructed_indices = reconstructed_indices_flat.reshape(shape)
    
    return tensor(reconstructed_indices, dtype=int32).unsqueeze(0)

def create_shares_legacy(latent, n, r, output_dir="shares"):
    """
    Legacy function for continuous latent space (for backward compatibility)
    This quantizes the latent values to [0, 255] range
    """
    if output_dir:
        if not os.path.exists(os.path.join(BASE_DIR, output_dir)):
            os.makedirs(os.path.join(BASE_DIR, output_dir))
    
    # Quantize latent to [0, 255] range
    latent_np = latent.detach().cpu().numpy()
    latent_min, latent_max = latent_np.min(), latent_np.max()
    latent_scaled = (latent_np - latent_min) / (latent_max - latent_min)
    latent_quantized = np.round(latent_scaled * 255).astype(np.uint8)
    
    shares, shares_extra = polynomial(latent_quantized, n=n, r=r)
    
    shares_with_positions = []
    for i, share in enumerate(shares):
        share_tensor = tensor(share, dtype=int32)
        position = (i + 1,)  
        shares_with_positions.append((share_tensor, position, shares_extra[i]))
        
        if output_dir:
            share_image = share.reshape(32, 32).astype(np.uint8)
            cv2.imwrite(os.path.join(BASE_DIR, output_dir, f"share_{i+1}.png"), share_image)
    
    return shares_with_positions

def combine_shares_legacy(shares_with_positions, r):
    """
    Legacy function for continuous latent space (for backward compatibility)
    """
    shares = np.array([share_tensor.numpy() for share_tensor, _, _ in shares_with_positions[:r]])
    shares_extra = [extra for _, _, extra in shares_with_positions[:r]]
    indices = [position[0] for _, position, _ in shares_with_positions[:r]]
    
    reconstructed_latent_quantized = decode(shares, shares_extra, indices, r=r)
    
    # Dequantize back to original range (approximate)
    reconstructed_latent = reconstructed_latent_quantized.astype(np.float32) / 255.0
    # Note: we lose the original min/max information, so this won't be exact
    # For better results, store min/max values separately
    
    return tensor(reconstructed_latent, dtype=float32)