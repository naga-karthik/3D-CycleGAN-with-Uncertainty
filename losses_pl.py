import torch
import scipy.ndimage as ndimage
import numpy as np

def gradient_consistency_loss(real_img, fake_img):
    """
    Recall that this is the only loss that calculates the loss across domains
    :param real_img: image from domain A
    :param fake_img: image from domain B
    :return: gradient consistency loss value
    """
    fake_img = fake_img.type(torch.cuda.FloatTensor)
    real_img = real_img.cpu().detach().numpy().squeeze()
    fake_img = fake_img.cpu().detach().numpy().squeeze()
    mean_loss_gc = []

    for i in range(real_img.shape[0]):

        if len(real_img.shape) != 3:
            real_img = real_img[i].squeeze()
            fake_img = fake_img[i].squeeze()

        gradx_img_a = ndimage.sobel(real_img, axis=0)  # axis=0 is the x-axis
        gradx_img_b = ndimage.sobel(fake_img, axis=0)
        ncc_x = normalized_cross_correlation(gradx_img_a, gradx_img_b)

        grady_img_a = ndimage.sobel(real_img, axis=1)  # axis=1 is the y-axis
        grady_img_b = ndimage.sobel(fake_img, axis=1)
        ncc_y = normalized_cross_correlation(grady_img_a, grady_img_b)

        gradz_img_a = ndimage.sobel(real_img, axis=2)  # axis=2 is the z-axis
        gradz_img_b = ndimage.sobel(fake_img, axis=2)
        ncc_z = normalized_cross_correlation(gradz_img_a, gradz_img_b)

        grad_corr_ab = 0.5 * (ncc_x + ncc_y + ncc_z)
        result = (1.0 - grad_corr_ab)
        mean_loss_gc.append(result)
   
    # print(len(mean_loss_gc))
    # print(torch.mean(torch.FloatTensor(mean_loss_gc)))
    return torch.mean(torch.Tensor(mean_loss_gc))


def normalized_cross_correlation(img_a, img_b):
    """
    Returns the normalized cross correlation between between two images
    """
    mu_a, sigma_a = np.mean(img_a), np.std(img_a)
    mu_b, sigma_b = np.mean(img_b), np.std(img_b)

    numerator = np.mean((img_a - mu_a) * (img_b - mu_b))
    denominator = sigma_a * sigma_b
    ncc_ab = numerator / (denominator + 1e-6)  # to avoid division by zero, in case

    return ncc_ab
