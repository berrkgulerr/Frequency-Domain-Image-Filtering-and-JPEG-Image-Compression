# Berk GÜLER 2310092
# Onur DEMİR 2309870

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.fftpack import dct
from PIL import Image

def part1(input_img_path, output_path):
    img = cv.imread(input_img_path, 0)
    (w, h) = img.shape
    bwt_filter = np.zeros((w, h))
    half_w, half_h = (w // 2), (h // 2)
    r = 80
    n = 3
    for u in range(w):
        for v in range(h):
            val = ((u - half_w) ** 2) + ((v - half_h) ** 2)
            if val != 0:
                x = pow(r / np.sqrt(val), 2 * n)
                bwt_filter[u][v] = 1 / (1 + x)

    img_ft = np.fft.fft2(img)
    img_ft_shifted = np.fft.fftshift(img_ft)
    img_filtered = np.multiply(img_ft_shifted, bwt_filter)
    image_inv_f_shift = np.fft.ifftshift(img_filtered)
    image_inv_f_trans = np.fft.ifft2(image_inv_f_shift)
    img_back = np.abs(image_inv_f_trans)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(output_path + '/edges.png', img_back)

    return img_back


# part1('./THE2-Images/1.png', './THE2-Images/edge_outputs')




def gaus_filter(w, h, r):
    filter = np.zeros([w, h])
    for u in range(w):
        for v in range(h):
            d_u_v = np.sqrt( ((u - int(w / 2)) ** 2) + ((v - int(h / 2)) ** 2))
            filter[u][v] = np.exp(-( (d_u_v**2) / (2*(r**2))))
    return filter

def butterworth_filter(w, h, r, n):
    filter = np.zeros([w, h])
    for u in range(w):
        for v in range(h):
            d_u_v = (((u - int(w / 2)) ** 2) + ((v - int(h / 2)) ** 2)) ** 0.5
            val = (( d_u_v / r) ** (2 * n))
            filter[u][v] = ((1.0 + val) ** (-1))
    return filter



def enhance_3(path_to_3, output_path):
    # read image
    img = cv.imread(path_to_3)
    blue, green, red = cv.split(img)

    # fourier transform the image
    img_ft_b = np.fft.fft2(blue)
    img_ft_g = np.fft.fft2(green)
    img_ft_r = np.fft.fft2(red)

    # flip the corners
    img_ft_shifted_b = np.fft.fftshift(img_ft_b)
    img_ft_shifted_g = np.fft.fftshift(img_ft_g)
    img_ft_shifted_r = np.fft.fftshift(img_ft_r)

    # gaussian filter
    gaussian_filt = gaus_filter(img.shape[0], img.shape[1], 50)

    # apply gaussian filter to image
    img_ft_filteredb = np.multiply(img_ft_shifted_b, gaussian_filt)
    img_ft_filteredg = np.multiply(img_ft_shifted_g, gaussian_filt)
    img_ft_filteredr = np.multiply(img_ft_shifted_r, gaussian_filt)

    # flip the corners back and inverse transform it
    img_denoisedb = np.fft.ifft2(np.fft.ifftshift(img_ft_filteredb))
    img_denoisedg = np.fft.ifft2(np.fft.ifftshift(img_ft_filteredg))
    img_denoisedr = np.fft.ifft2(np.fft.ifftshift(img_ft_filteredr))

    img_denoisedb = np.abs(img_denoisedb)
    img_denoisedg = np.abs(img_denoisedg)
    img_denoisedr = np.abs(img_denoisedr)

    # merge the image
    out = cv.merge([img_denoisedb, img_denoisedg, img_denoisedr])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(output_path + '/enhanced.png', out)

# enhance_3('./THE2-Images/3.png', './THE2-Images/enhance3_outputs')




def enhance_4(path_to_4, output_path):
    # read image
    img = cv.imread(path_to_4)
    blue, green, red = cv.split(img)

    # fourier transform the image
    img_ft_b = np.fft.fft2(blue)
    img_ft_g = np.fft.fft2(green)
    img_ft_r = np.fft.fft2(red)

    # flip the corners
    img_ft_shifted_b = np.fft.fftshift(img_ft_b)
    img_ft_shifted_g = np.fft.fftshift(img_ft_g)
    img_ft_shifted_r = np.fft.fftshift(img_ft_r)

    # gaussian filter
    gaussian_filt = butterworth_filter(img.shape[0], img.shape[1], 50, 0.5)

    # apply gaussian filter to image
    img_ft_filteredb = np.multiply(img_ft_shifted_b, gaussian_filt)
    img_ft_filteredg = np.multiply(img_ft_shifted_g, gaussian_filt)
    img_ft_filteredr = np.multiply(img_ft_shifted_r, gaussian_filt)

    # flip the corners back and inverse transform it
    img_denoisedb = np.fft.ifft2(np.fft.ifftshift(img_ft_filteredb))
    img_denoisedg = np.fft.ifft2(np.fft.ifftshift(img_ft_filteredg))
    img_denoisedr = np.fft.ifft2(np.fft.ifftshift(img_ft_filteredr))

    img_denoisedb = np.abs(img_denoisedb)
    img_denoisedg = np.abs(img_denoisedg)
    img_denoisedr = np.abs(img_denoisedr)

    # merge the image
    out = cv.merge([img_denoisedb, img_denoisedg, img_denoisedr])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(output_path + '/enhanced.png', out)
# enhance_4('./THE2-Images/4.png', './THE2-Images/enhance4_outputs')


def the2_write (input_img_path , output_path ):
    # Read the input image
    img = cv.imread(input_img_path)
    h,w,u = img.shape

    # Convert color space to YCrCb
    img_YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

    # Split the image to Y-Cr-Cb channels
    y, cr, cb = cv.split(img_YCrCb)

    # downsample cr and cb [Y:Cr:Cb] = [4:2:2]
    crsub = cr[::2,::2] 
    cbsub = cb[::2,::2]

    # Quantization Table for Y channel
    q_y=np.array([[16,11,10,16,24,40,51,61],
        [12,12,14,19,26,48,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]])

    # Quantization table for Chrominance channels
    q_c=np.array([[17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]])

    # Get row and col size of Y Channel
    y_rows = y.shape[0]
    y_cols = y.shape[1]
    # create temp np array for dct of Y Channel
    y_dct = np.zeros((y_rows, y_cols))
    # create temp np array of Y channel
    y_temp = np.zeros((y_rows, y_cols))
    y_temp[:y_rows, : y_cols] = y
    # substract 128 in order to get range [-128, ...127]
    y_temp = y_temp-128
    # Get blocks of size 8
    block_size_v = y_rows//8
    block_size_h = y_cols//8
    for row in range(block_size_v):
        for col in range(block_size_h):
            # take DCT of 8x8 block
            sub_img = cv.dct(y_temp[row*8 : (row+1)*8, col*8 : (col+1)*8])
            # save sub_img to y_dct after quantization
            y_dct[row*8:(row+1)*8, col*8 : (col+1)*8] = np.round(sub_img/q_y)
    # append y_dct to compressed_img array

    # Do the same operations that are done in Y channel for Cr Channel
    cr_rows = crsub.shape[0]
    cr_cols = crsub.shape[1]
    cr_dct = np.zeros((cr_rows, cr_cols))
    cr_temp = np.zeros((cr_rows, cr_cols))
    cr_temp[:cr_rows, : cr_cols] = crsub
    cr_temp = cr_temp-128
    block_size_v = cr_rows//8
    block_size_h = cr_cols//8
    for row in range(block_size_v):
        for col in range(block_size_h):
            sub_img = cv.dct(cr_temp[row*8 : (row+1)*8, col*8 : (col+1)*8])
            cr_dct[row*8:(row+1)*8, col*8 : (col+1)*8] = np.round(sub_img/q_c)

    # Do the same operations that are done in Y channel for Cb Channel
    cb_rows = cbsub.shape[0]
    cb_cols = cbsub.shape[1]
    cb_dct = np.zeros((cb_rows, cb_cols))
    cb_temp = np.zeros((cb_rows, cb_cols))
    cb_temp[:cb_rows, : cb_cols] = cbsub
    cb_temp = cb_temp-128
    block_size_v = cb_rows//8
    block_size_h = cb_cols//8
    for row in range(block_size_v):
        for col in range(block_size_h):
            sub_img = cv.dct(cb_temp[row*8 : (row+1)*8, col*8 : (col+1)*8])
            cb_dct[row*8:(row+1)*8, col*8 : (col+1)*8] = np.round(sub_img/q_c)

    #### Decompression ####
    recompressed_img = np.zeros((h,w,3), np.uint8)
    y_rows = y_dct.shape[0]
    y_cols = y_dct.shape[1]
    block_size_v = y_rows//8
    block_size_h = y_cols//8
    y_idct = np.zeros((y_rows,y_cols), np.uint8)
    for row in range(block_size_v):
        for col in range(block_size_h):
                            dequant_sub_img=y_dct[row*8:(row+1)*8,col*8:(col+1)*8]*q_y
                            idtc_dequant_sub_img = np.round(cv.idct(dequant_sub_img))+128
                            idtc_dequant_sub_img[idtc_dequant_sub_img>255]=255
                            idtc_dequant_sub_img[idtc_dequant_sub_img<0]=0
                            y_idct[row*8:(row+1)*8,col*8:(col+1)*8]=idtc_dequant_sub_img
    resized_idtc_dequant_sub_img=cv.resize(y_idct,(w,h))
    recompressed_img[:,:,0]=np.round(resized_idtc_dequant_sub_img)

    cr_rows = cr_dct.shape[0]
    cr_cols = cr_dct.shape[1]
    block_size_v = cr_rows//8
    block_size_h = cr_cols//8
    cr_idct = np.zeros((cr_rows,cr_cols), np.uint8)
    for row in range(block_size_v):
        for col in range(block_size_h):
                            dequant_sub_img=cr_dct[row*8:(row+1)*8,col*8:(col+1)*8]*q_c
                            idtc_dequant_sub_img = np.round(cv.idct(dequant_sub_img))+128
                            idtc_dequant_sub_img[idtc_dequant_sub_img>255]=255
                            idtc_dequant_sub_img[idtc_dequant_sub_img<0]=0
                            cr_idct[row*8:(row+1)*8,col*8:(col+1)*8]=idtc_dequant_sub_img
    resized_idtc_dequant_sub_img=cv.resize(cr_idct,(w,h))
    recompressed_img[:,:,1]=np.round(resized_idtc_dequant_sub_img)

    cb_rows = cb_dct.shape[0]
    cb_cols = cb_dct.shape[1]
    block_size_v = cb_rows//8
    block_size_h = cb_cols//8
    cb_idct = np.zeros((cb_rows,cb_cols), np.uint8)
    for row in range(block_size_v):
        for col in range(block_size_h):
                            dequant_sub_img=cb_dct[row*8:(row+1)*8,col*8:(col+1)*8]*q_c
                            idtc_dequant_sub_img = np.round(cv.idct(dequant_sub_img))+128
                            idtc_dequant_sub_img[idtc_dequant_sub_img>255]=255
                            idtc_dequant_sub_img[idtc_dequant_sub_img<0]=0
                            cb_idct[row*8:(row+1)*8,col*8:(col+1)*8]=idtc_dequant_sub_img
    resized_idtc_dequant_sub_img=cv.resize(cb_idct,(w,h))
    recompressed_img[:,:,2]=np.round(resized_idtc_dequant_sub_img)

    compressed_image=cv.cvtColor(recompressed_img, cv.COLOR_YCrCb2BGR)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    path = output_path + '/compressed_img.jpeg'
    cv.imwrite(path, compressed_image)
    original_size = os.path.getsize(input_img_path)
    compressed_size = os.path.getsize(path)
    compression_ratio = original_size/compressed_size
    print(original_size, compressed_size, compression_ratio)
    return path


def the2_read(input_img_path):
    image = cv.imread(input_img_path)
    cv.imshow("compressed_image", image)
    cv.waitKey(0)
    cv.destroyAllWindows() 

# the2_read(the2_write('./THE2-Images/5.png', './THE2-Images/compression_outputs'))