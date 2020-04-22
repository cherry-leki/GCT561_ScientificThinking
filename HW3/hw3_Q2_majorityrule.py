import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def addNoise(image, prob):
    tmpImg = np.zeros(image.shape)

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            noise = np.random.choice([255, 0], p=[prob, 1-prob])
            tmpImg[i][j] = abs(noise - image[i][j])

    return tmpImg


def findErrorCorrection(images):
    tmpImg = np.zeros(images[0].shape)

    for i in range(0, tmpImg.shape[0]):
        for j in range(0, tmpImg.shape[1]):
            nums = [images[0][i][j], images[1][i][j], images[2][i][j]]
            value = 0 if nums.count(0) > 1 else 255
            tmpImg[i][j] = value

    return tmpImg


''' Initialize image and noise probability '''
noise_prob = 0.1
img = cv2.imread('spritethinker.png', 0)
# print(img)


''' Create noise image '''
# noiseImg1 = cv2.bitwise_not(img)      # 255-img
noiseImg1 = addNoise(img, noise_prob)
noiseImg2 = addNoise(img, noise_prob)
noiseImg3 = addNoise(img, noise_prob)


''' Create error corrected image '''
ecImage = findErrorCorrection([noiseImg1, noiseImg2, noiseImg3])

''' Save noise images'''
if not os.path.exists("result"):
    os.mkdir("result")
cv2.imwrite("result/noiseImg1.png", noiseImg1)
cv2.imwrite("result/noiseImg2.png", noiseImg2)
cv2.imwrite("result/noiseImg3.png", noiseImg3)
cv2.imwrite("result/ecImage.png", ecImage)


''' Show images'''
fig, axes = plt.subplots(3, 3)
# Original image
ax0 = plt.subplot2grid((3, 3), (0, 0), rowspan=3)
ax0.imshow(img, 'gray')
ax0.set_title('Original')
ax0.axis('off')
# Noise images
axes[0][1].imshow(noiseImg1, 'gray')
axes[0][1].set_title('Noise image 1')
axes[1][1].imshow(noiseImg2, 'gray')
axes[1][1].set_title('Noise image 2')
axes[2][1].imshow(noiseImg3, 'gray')
axes[2][1].set_title('Noise image 3')
# Error corrected image
ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
ax2.imshow(ecImage, 'gray')
ax2.set_title('EC image')
ax2.axis('off')

for ax in axes.flat:
    ax.axis('off')
    ax.axes.get_yaxis().set_visible(False)

plt.tight_layout(h_pad=None)
plt.savefig(fname='result/resultall.png', bbox_inches='tight', pad_inches=0.1)    # save result image
plt.show()


