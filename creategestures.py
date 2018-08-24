# cv2 - real time computer vision library
import cv2
import numpy as np
import os

# dimensions of image
image_x, image_y = 50, 50

# video capture using default(0) camera - global
cap = cv2.VideoCapture(0)

# Gaussian Mixture-based Background/Foreground Segmentation Algorithm for motion analysis
# The weights of the mixture represent the time proportions that those colours stay in the scene.
# The probable background colours are the ones which stay longer and more static
# Parameters -
# history	Length of the history.
# varThreshold	Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a
# pixel is well described by the background model. This parameter does not affect the background update.
# detectShadows	If true, the algorithm will detect shadows and mark them.
# It decreases the speed a bit, so if you do not need this feature, set the parameter to false.
fbag = cv2.createBackgroundSubtractorMOG2()

# create folder if it does not exist
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

# Parameters -
# g_id - gesture name
def main(g_id):

    # number of snaps to be captured
    total_pics = 1200
    # locally declared VideoCapture
    cap = cv2.VideoCapture(0)

    # x, y - starting indices of frame
    # h, w - height and width of frame
    x, y, w, h = 300, 50, 350, 350

    # create folder with g_id as label
    create_folder("gestures/" + str(g_id))
    # counter for pictures
    pic_no = 0
    # press c to set this bool to true and begin capturing
    flag_start_capturing = False
    frames = 0

    while True:
        # next frame in camera via cap
        # ret - return value from camera frame (true / false)
        ret, frame = cap.read()
        # flip frame vertically - 1, horizontally - 0
        frame = cv2.flip(frame, 1)
        # hsv - hue saturation value
        # change pic from one color space to another
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # create mask which captures colors with pixel intensities in the range of rgb(60,50,2) to rgb(255,150,25)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
        # bit AND frame and mask
        res = cv2.bitwise_and(frame, frame, mask=mask2)

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
            # contourArea must be grateer than 10000 and frames are more than 50
            if cv2.contourArea(contour) > 10000 and frames > 50:
                # find the most prominent contour in the image and crop it to the bounding rectangle
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                # increment pic_no
                pic_no += 1
                # crop the frame
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                # enter border if h and w of frame do not match
                if w1 > h1:
                    # copyMakeBorder( src, dst, top, bottom, left, right, borderType, value)
                    save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                # fill all pics to 50 x 50 pixel size
                save_img = cv2.resize(save_img, (image_x, image_y))

                # display text on the frame
                # image
                # Text data
                # Position coordinates - bottom-left corner
                # Font type
                # Font Scale
                # regular things like color, thickness, lineType etc
                cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))

                # writing the captured image to gestures/g_id/pic_no.jpg
                cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

        # Parameters -
        # image
        # top left
        # bottom right
        # color code
        # line type
        # mark the bounding rectangle of capture frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", frame)
        cv2.imshow("thresh", thresh)

        # wait for user keyboard input
        # delay - 1 ms delay
        keypress = cv2.waitKey(1)

        # if key press is 'c', start capture
        if keypress == ord('c'):
            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing == True:
            frames += 1
        if pic_no == total_pics:
            break


g_id = input("Enter gesture number: ")
main(g_id)