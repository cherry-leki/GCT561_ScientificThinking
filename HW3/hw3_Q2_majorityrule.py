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


def computePixelProb(image):
    black = 0
    white = 0

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i][j] == 0:
                black = black + 1
            else:
                white = white + 1

    p_0 = black / (image.shape[0] * image.shape[1])
    p_1 = white / (image.shape[0] * image.shape[1])

    print("Counting pixels (black/white) : " + str(black) + " / " + str(white))

    return [p_0, p_1]


def comparePixels(original, image):
    error = 0
    correct = 0

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if original[i][j] == image[i][j]:
                correct = correct + 1
            else:
                error = error + 1

    p_crt = correct / (image.shape[0] * image.shape[1])
    p_err = error / (image.shape[0] * image.shape[1])

    print("Counting errors (correct/error) : " + str(correct) + ", " + str(error))

    return [p_crt, p_err]


''' Initialize image and noise probability '''
noise_prob = 0.1
img = cv2.imread('spritethinker.png', 0)
# print(img)
print("### Original image ###")
print("P(0), P(1): " + str(computePixelProb(img)) + '\n')

''' Create noise image '''
# noiseImg1 = cv2.bitwise_not(img)      # 255-img
noiseImg1 = addNoise(img, noise_prob)
noiseImg2 = addNoise(img, noise_prob)
noiseImg3 = addNoise(img, noise_prob)

print("### Noise image 1 ###")
print("P_correct, P_error: " + str(comparePixels(img, noiseImg1)) + '\n')
print("### Noise image 2 ###")
print("P_correct, P_error: " + str(comparePixels(img, noiseImg2)) + '\n')
print("### Noise image 3 ###")
print("P_correct, P_error: " + str(comparePixels(img, noiseImg3)) + '\n')


''' Create error corrected image '''
ecImage = findErrorCorrection([noiseImg1, noiseImg2, noiseImg3])

print("### Error corrected image  ###")
print("P_correct, P_error: " + str(comparePixels(img, ecImage)) + '\n')


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


