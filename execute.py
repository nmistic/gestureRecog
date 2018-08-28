import cv2
from keras.models import load_model
import numpy as np
import os

# load from model.h5 file
model = load_model('model.h5')


def main():
    emojis = get_emojis()
    cap = cv2.VideoCapture(0)

    # x, y - starting indices of frame
    # h, w - height and width of frame
    x, y, w, h = 300, 50, 350, 350

    # cap - default camera object, while opened
    while cap.isOpened():

        # return value and image from cap
        ret, img = cap.read()

        # flip frame vertically - 1, horizontally - 0
        img = cv2.flip(img, 1)

        # hsv - hue saturation value
        # change pic from one color space to another
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # create mask which captures colors with pixel intensities in the range of rgb(60,50,2) to rgb(255,150,25)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))

        # bit AND frame and mask
        res = cv2.bitwise_and(img, img, mask=mask2)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # blur filter for image using Gaussian function
        # specify the width and height of the kernel which should be positive and odd.
        # We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively.
        # If only sigmaX is specified, sigmaY is taken as equal to sigmaX.
        # If both are given as zeros, they are calculated from the kernel size.
        # Gaussian filtering is highly effective in removing Gaussian noise from the image.
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        # structuring element or kernel which decides the nature of operation in dilation
        kernel_square = np.ones((5, 5), np.uint8)

        # a pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’.
        # So it increases the white region in the image or size of foreground object increases.
        dilation = cv2.dilate(median, kernel_square, iterations=2)

        # difference between dilation and erosion of an image
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)

        # First argument is the source image, which should be a grayscale image.
        # Second argument is the threshold value which is used to classify the pixel values.
        # Third argument is the maxVal which represents the value to be given if pixel value is more than
        # (sometimes less than) the threshold value
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        thresh = thresh[y:y + h, x:x + w]

        # first one is source image, second is contour retrieval mode, third is contour approximation method.
        # And it outputs the contours and hierarchy. contours is a Python list of all the contours in the image.
        # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

        if len(contours) > 0:
            # find max contour on basis on contourArea
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:

                # find bounding rectangle of the contour
                x, y, w1, h1 = cv2.boundingRect(contour)

                # pick up the contour area from the image
                newImage = thresh[y:y + h1, x:x + w1]

                # resize this image to fit into our model dimensions
                newImage = cv2.resize(newImage, (50, 50))

                # predict the class for the contour
                pred_probab, pred_class = keras_predict(model, newImage)
                print(pred_class, pred_probab)

                # display the image at the predicted class index
                img = overlay(img, emojis[pred_class], 400, 250, 90, 90)

        x, y, w, h = 300, 50, 350, 350
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


def keras_predict(model, image):
    # reshape image dimensions
    processed = keras_process_image(image)

    # predict probabilities of classes
    pred_probab = model.predict(processed)[0]

    # get the index of the class with the maximum predict probability
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


# pre process image dimensions for predictions
def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


# return a vector of all emojis stored in emojis_folder
def get_emojis():
    emojis_folder = 'hand_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png', -1))
    return emojis


def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    # Grab the BRG planes
    overlay_img = overlay_t_img[:, :, :3]

    # And the alpha plane
    overlay_mask = overlay_t_img[:, :, 3:]

    # calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel to use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
main()

