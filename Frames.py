# Extract all frames from a video using Python (OpenCV)
import cv2
def videoRead(video, name):
    cap = cv2.VideoCapture(video)
    img_index = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        #test values, please add here the PATH where you want to save the frames
        cv2.imwrite('B10/'+str(name)+'_' + str(img_index) + '.jpg', frame)
        img_index += 1
    cap.release()
    cv2.destroyAllWindows()


