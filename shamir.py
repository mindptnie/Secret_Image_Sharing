#>250 補回原來數值 
import time
import numpy as np
import argparse
import png
import sys
import os
from torch import tensor
from torch import int32
from torch import float32
from torch import randn
import cv2
from PIL import Image
from Crypto.Util.number import math, inverse

MODULO = 251  # 使用 mod 251

# Root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 處理像素值超過 250 的問題
def clamp_pixel_values(img):
    img = np.where(img > 250, 250, img)
    return img

def preprocessing(path):
    img = Image.open(path)
    data = np.asarray(img)
    return data.flatten(),data.shape

def insert_text_chunk(src_png, dst_png, text):
    '''在png中的第二个chunk插入自定义内容'''
    reader = png.Reader(filename=src_png)
    chunks = reader.chunks()#创建一个每次返回一个chunk的生成器
    chunk_list = list(chunks)
    chunk_item = tuple([b'tEXt', text])

    index = 1
    chunk_list.insert(index, chunk_item)

    with open(dst_png, 'wb') as dst_file:
        png.write_chunks(dst_file, chunk_list)

def read_text_chunk(src_png, index=1):
    '''读取png的第index个chunk'''
    reader = png.Reader(filename=src_png)
    chunks = reader.chunks()
    chunk_list = list(chunks)
    img_extra = chunk_list[index][1].decode()
    img_extra = eval(img_extra)
    return img_extra

def polynomial(img, n, r):
    img = clamp_pixel_values(img)
    num_pixels = img.shape[0]
    coefficients = np.random.randint(low=0, high=MODULO, size=(num_pixels, r - 1))
    secret_imgs = []
    imgs_extra = []

    for i in range(1, n + 1):
        base = np.array([i ** j for j in range(1, r)])
        base = np.matmul(coefficients, base)
        secret_img = (img + base) % MODULO

        indices = np.where(secret_img > 250)[0]
        img_extra = [(int(idx), int(secret_img[idx])) for idx in indices]
        secret_img[indices] = 250  # 超過 250 的改為 250

        secret_imgs.append(secret_img)
        imgs_extra.append(img_extra)

    return np.array(secret_imgs), imgs_extra

def polynomial_group(img, n, r):
    assert r == 3, "目前只支援 r=3 (即 aX² + bX + c)"
    img = clamp_pixel_values(img)
    
    # 每三個 latent 值為一組係數 (a, b, c)
    groups = []
    for i in range(0, len(img), 3):
        group = img[i:i+3]
        if len(group) < 3:
            # 若最後一組不足 3 個元素，補 0
            group = np.pad(group, (0, 3 - len(group)), 'constant')
        groups.append(group)
    groups = np.array(groups)  # shape: [342, 3]

    secret_imgs = []
    imgs_extra = []

    for i in range(1, n + 1):
        x = i
        x_powers = np.array([x**2, x, 1])  # [x², x, 1]
        shares = np.dot(groups, x_powers) % MODULO
        shares = shares.flatten()

        # 補償處理（clamp >250 為 250，並記錄原值）
        indices = np.where(shares > 250)[0]
        img_extra = [(int(idx), int(shares[idx])) for idx in indices]
        shares[indices] = 250

        secret_imgs.append(shares)
        imgs_extra.append(img_extra)

    return np.array(secret_imgs), imgs_extra

def format_size(size_bytes):
    """ 根据字节大小自动调整单位 """
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def get_file_size(file_path):
    """ 获取文件大小并格式化输出 """
    try:
        size = os.path.getsize(file_path)
        return format_size(size)
    except OSError as e:
        return f"Error: {e}"

def lagrange(x, y, num_points, x_test):
    l = np.zeros(shape=(num_points,))
    for k in range(num_points):

        l[k] = 1
        for k_ in range(num_points):

            if k != k_:
                d = int(x[k] - x[k_])
                inv_d = inverse(d, MODULO)
                l[k] = l[k] * (x_test - x[k_]) * inv_d % MODULO

            else:
                pass
    L = 0
    for i in range(num_points):
        L += y[i] * l[i]
    return L

def decode(imgs, imgs_extra, index, r):
    assert imgs.shape[0] >= r
    x = np.array(index)
    dim = imgs.shape[1]
    img = []
    last_percent_reported = None

    # ✅ 正確補償：先補回原始值
    for i in range(r):
        for idx, original in imgs_extra[i]:
            imgs[i][idx] = original

    for i in range(dim):
        y = imgs[:, i]
        pixel = lagrange(x, y, r, 0) % MODULO
        img.append(pixel)

        # 進度條
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

def compare_images(image1_path, image2_path):
    image1 = np.array(Image.open(image1_path))
    image2 = np.array(Image.open(image2_path))
    diff = np.abs(image1 - image2)
    diff_value = round(np.mean(diff), 4)
    print("Mean difference:", diff_value)
    print("Max difference:", round(np.max(diff), 4))
    print("Min difference:", round(np.min(diff), 4))
    print("Standard deviation of difference:", round(np.std(diff), 4))

def quantize_latent(latent, min_val=-3, max_val=3):
    latent_np = latent.detach().cpu().numpy()
    latent_scaled = (latent_np - min_val) / (max_val - min_val)
    latent_quantized = np.round(latent_scaled * 255).astype(np.uint8)
    return latent_quantized

def dequantize_latent(latent_quantized, min_val=-3, max_val=3):
    latent_scaled = latent_quantized.astype(np.float32) / 255.0
    latent = latent_scaled * (max_val - min_val) + min_val
    return latent

def create_shares(latent, n, r, output_dir="shares", group_polynomial=False):
    if not os.path.exists(os.path.join(BASE_DIR,output_dir)):
        os.makedirs(os.path.join(BASE_DIR,output_dir))
        
    latent_quantized = quantize_latent(latent)
    if group_polynomial:
        shares, shares_extra = polynomial_group(latent_quantized, n=n, r=r)
    else:
        shares, shares_extra = polynomial(latent_quantized, n=n, r=r)

    shares_with_positions = []
    for i, share in enumerate(shares):
        share_tensor = tensor(share, dtype=int32)
        position = (i + 1,)  
        shares_with_positions.append((share_tensor, position, shares_extra[i]))

        share_image = share.reshape(32, 32).astype(np.uint8)

        cv2.imwrite(os.path.join(BASE_DIR,output_dir, f"share_{i+1}.png"), share_image)

    #return shares_with_positions
    return shares_with_positions

def combine_shares(shares_with_positions, r):
    shares = np.array([share_tensor.numpy() for share_tensor, _, _ in shares_with_positions[:r]])
    shares_extra = [extra for _, _, extra in shares_with_positions[:r]]
    indices = [position[0] for _, position, _ in shares_with_positions[:r]]

    reconstructed_latent_quantized = decode(shares, shares_extra, indices, r=r)
    reconstructed_latent = dequantize_latent(reconstructed_latent_quantized)

    return tensor(reconstructed_latent, dtype=float32)

def main():
    parser = argparse.ArgumentParser(description='Shamir Secret Image Sharing')
    parser.add_argument('-e', '--encode', help='Path to the image to be encoded')
    parser.add_argument('-d', '--decode', help='Path for the origin image to be saved')
    parser.add_argument('-n', type=int, help='The total number of shares')
    parser.add_argument('-r', type=int, help='The threshold number of shares to reconstruct the image')
    parser.add_argument('-i', '--index', nargs='+', type=int, help='The index of shares to use for decoding')
    parser.add_argument('-c', '--compare', nargs=2, help='Compare two images')
    args = parser.parse_args()

    if args.encode:
        start_time = time.time()
        print("\n=== Starting image encoding process ===")

        if not args.r:
            print("Error: Threshold number 'r' is required for decoding")
            return
        if not args.n:
            print("Error: Total number 'n' of shares is required for decoding")
            return
        if args.r > args.n:
            print("Error: Threshold 'r' cannot be greater than the total number 'n' of shares")
            return

        img_flattened, shape = preprocessing(args.encode)
        secret_imgs, imgs_extra = polynomial(img_flattened, n=args.n, r=args.r)
        to_save = secret_imgs.reshape(args.n, *shape)
        for i, img in enumerate(to_save):
            secret_img_path = f"secret_{i + 1}.png"
            Image.fromarray(img.astype(np.uint8)).save(secret_img_path)
            img_extra = str(list((imgs_extra[i]))).encode()
            insert_text_chunk(secret_img_path, secret_img_path, img_extra)
            size = get_file_size(secret_img_path)
            print(f"{secret_img_path} saved.",size)
            

        end_time = time.time()
        print("=== Image encoding completed. Time elapsed: {:.2f} seconds ===".format(end_time - start_time))

    if args.decode:
        start_time = time.time()
        print("\n=== Starting image decoding process ===")

        if not args.r:
            print("Error: Threshold number 'r' is required for decoding")
            return

        input_imgs = []
        input_imgs_extra = []
        for i in args.index:
            secret_img_path = f"secret_{i}.png"
            img_extra = read_text_chunk(secret_img_path)
            img, shape = preprocessing(secret_img_path)
            input_imgs.append(img)
            input_imgs_extra.append(img_extra)
        input_imgs = np.array(input_imgs)
        origin_img = decode(input_imgs, input_imgs_extra, args.index, r=args.r)
        origin_img = origin_img.reshape(*shape)
        Image.fromarray(origin_img.astype(np.uint8)).save(args.decode)
        size = get_file_size(args.decode)
        print(f"{args.decode} saved.",size)

        end_time = time.time()
        print("=== Image decoding completed. Time elapsed: {:.2f} seconds ===".format(end_time - start_time))

    if args.compare:
        print("\n=== Starting image comparison ===")

        compare_images(args.compare[0], args.compare[1])

        print("=== Image comparison completed.  ===")

if __name__ == "__main__":
    main()