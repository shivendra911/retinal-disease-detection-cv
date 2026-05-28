import cv2
import numpy as np

def crop_image_from_gray(img, tol=7):
    """
    Crop the black borders from fundus images.
    """
    if len(img.shape) == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def ben_graham_clahe(img):
    """
    Compute Ben Graham's preprocessing using CLAHE.
    """
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0) ,  10), -4 ,128)
    return image
