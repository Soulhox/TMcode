import cv2
import numpy as np
import pytesseract
from scipy.spatial import distance
import statistics as stat
import time
import pandas as pd
import re
import operator
from time import sleep
import matplotlib.pyplot as plt
from detecto import core, utils, visualize

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# TEST ROUTES AND MODEL, PLEASE HERE TYPE THE PATH AND ROUTES TO YOUR MODEL
model = core.Model.load('Modelo.pth', [
                        '8', 'F14', 'F28', 'G11', 'G12', 'G44', 'H13', 'H75', 'J74', 'K16', 'L18'])
"*************************************DICCIONARIES************************************"
fromSymbolToNumber = {
    'a': '',
    'b': '',
    'c': '0',
    'd': '0',
    'e': '0',
    'f': '1',
    'g': '6',
    'h': '',
    'i': '1',
    'j': '7',
    'k': '',
    'l': '1',
    'm': '',
    'n': '',
    'o': '0',
    'p': '0',
    'q': '0',
    'r': '',
    's': '5',
    't': '7',
    'u': '',
    'v': '',
    'w': '',
    'x': '',
    'y': '',
    'z': '',
    'A': '4',
    'B': '8',
    'C': '0',
    'D': '0',
    'E': '3',
    'F': '1',
    'G': '6',
    'H': '',
    'I': '1',
    'J': '7',
    'K': '',
    'L': '1',
    'M': '',
    'N': '',
    'O': '0',
    'P': '',
    'Q': '0',
    'R': '',
    'S': '5',
    'T': '7',
    'U': '',
    'V': '0',
    'W': '',
    'X': '',
    'Y': '',
    'Z': '',
    ']': '1',
    '[': '1',
    '(': '0',
    '|': '1',
    ')': '1',
    '/': '7',
    '()': '0',
    '}': '1',
    '': ''
}
fromNumberToLetter = {
    '1': 'T',
    '2': '',
    '3': 'B',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': '',
    '0': 'O',
    ']': 'T',
    '[': 'T',
    '(': '',
    '|': 'I',
    ')': 'I',
    '/': 'I',
    'y': 'T',
    '': '',
    '}': 'I'
}
"""***************************************************************************
# NAME: min_euclidean
# DESCRIPTION: Calculates the shortest euclidean distance in the set of coordinates.
#
# PARAMETERS: organized: organized set of coordinates previously filtered
                by aspect relation
#
# RETURNS:    last_distance: last shortest distance calculated.
#
#
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  03/Feb/2021   Juan Sebastian Corredor
***************************************************************************"""


def min_euclidean(organized):
    last_distance = 50
    for i in range(len(organized)-1):
        for j in range(i+1, len(organized)):
            [x1, y1, w1, h1] = [organized[i][0], organized[i]
                                [1], organized[i][2], organized[i][3]]
            [x2, y2, w2, h2] = [organized[j][0], organized[j]
                                [1], organized[j][2], organized[j][3]]
            u = (x1, y1)
            v = (x2, y2)
            euc_distance = distance.euclidean(u, v)
            if (euc_distance < last_distance) and euc_distance != 0:
                last_distance = euc_distance
    return last_distance


"""***************************************************************************
# NAME: Cleaner
# DESCRIPTION: Cleans the final text of the bus number
#
# PARAMETERS: InputText: the text grouped previously that has to be cleansed
#
# RETURNS:    Clean_text: The text cleansed and ready to print
#
#
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  05/Feb/2021   Juan Sebastian Corredor
***************************************************************************"""


def Cleaner(InputText):
    clean_text = ''
    InputText = InputText.replace("!", '1')
    InputText = re.sub("[^a-zA-Z0-9(\)[\]\}/|]", '', InputText)
    InputText = InputText.replace("()", '0')
    for i in range(len(InputText)):
        new_text = InputText[i]
        if(i == 0):
            if new_text.isalpha():
                new_text = re.sub("y", 'T', new_text)
                new_text = re.sub("R", 'B', new_text)
                new_text = re.sub("q", 'T', new_text)
                clean_text += new_text
            else:
                clean_text += fromNumberToLetter[new_text]
        else:
            if new_text.isdigit():
                clean_text += new_text
            else:
                clean_text += fromSymbolToNumber[new_text]
    if len(clean_text) < 4:
        return 'Error'
    else:
        clean_text = re.sub("[^a-zA-Z0-9]", '', clean_text)
        return clean_text


"""***************************************************************************
# NAME: Checker
# DESCRIPTION: Checks that the input text is one letter plus 4 or 5 numbers format
#
# PARAMETERS: bus_number: The bus number already cleansed
#
# RETURNS:    Checked_text: returns only the text in the format mentioned above
#               other text will be erased
#
#
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  09/Feb/2021   Juan Sebastian Corredor
***************************************************************************"""


def Checker(bus_number):
    if len(bus_number) >= 4 and len(bus_number) < 6:
        for i in range(len(bus_number)):
            if i == 0:
                if bus_number[i].isalpha():
                    continue
                else:
                    return 'Error'
            else:
                if bus_number[i].isdigit():
                    continue
                else:
                    return 'Error'
        return bus_number
    else:
        return 'Error'


"""***************************************************************************
# NAME: VideoRead
# DESCRIPTION: Read the incoming video and returns the bus number and route
#
# PARAMETERS: Video: the video that contains a bus
#
# RETURNS:    BusNumbers: returns a list with all the readings made
#             FilteredNumbers: returns a list with only the potential bus numbers
#
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  04/03/2021      Juan Sebastian Corredor Angarita
***************************************************************************"""


def VideoRead(video):
    FilteredNumbers = []
    error_count = 0
    detected = 0
    read = cv2.VideoCapture(video)
    analized = ''
    RouteNumbers = []
    while(read.isOpened()):
        ret, original_frame = read.read()
        if ret == False:
            break
        bus_number = get_number(original_frame)
        if(bus_number == ''):
            bus_number = "Error"
        if bus_number == 'Error' and detected == 1:
            error_count += 1
            if error_count >= 50:
                analized = AnalizeCharacters(FilteredNumbers)
                FilteredNumbers = []
                error_count = 0
                detected = 0
                try:
                    print('BUS DETECTED: ', analized, stat.mode(RouteNumbers))
                    numbers.append([stat.mode(RouteNumbers), analized])
                except Exception:
                    continue
                NUMBERS = pd.DataFrame(numbers, columns=['number', 'read'])
                NUMBERS.to_csv('OUTPUT.csv')
        if bus_number != 'Error':
            detected = 1
            error_count = 0
            predictions = model.predict(original_frame)
            labels, boxes, scores = predictions
            labels = np.asarray(labels)
            try:
                FilteredNumbers.append(bus_number.upper())
                RouteNumbers.append(labels[0])
            except IndexError:
                continue
    read.release()
    return analized


"""***************************************************************************   
# NAME: get_number
# DESCRIPTION: detect and read the bus number and route from the video
#
# PARAMETERS: Original_frame: the frame from the video that has to be 
#             read
#                
# RETURNS:    bus_number: the number detected and converted to string
#  
#             
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  09/Feb/2021   Juan Sebastian Corredor Angarita   
***************************************************************************"""


def get_number(original_frame):
    new_frame = original_frame.copy()
    posible_characters = []
    characters = []
    filtered_rectangles = []
    filtered_coordinates = []
    final_boxes = []
    mode_samples = []
    mode_error = 5
    final_text = ''
    grouped_text = ''
    # ALGORITHM STARTS HERE
    gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    ret3, th3 = cv2.threshold(
        gray_frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    dilated = cv2.dilate(th3, kernel, iterations=0)
    # FIND CONTOURS
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # DRAW CONTOURS (with random colors)
    rand_color = np.uint8(np.random.randint(255, size=(len(contours), 3)))
    all_contours = original_frame.copy()
    for f in range(len(contours)):
        cv2.drawContours(all_contours, contours, f,
                         (int(rand_color[f, 0]), int(rand_color[f, 1]),
                          int(rand_color[f, 2])), 2)
    for f in range(len(contours)):
        [x, y, w, h] = cv2.boundingRect(contours[f])
        area = cv2.contourArea(contours[f])
        aspect = w/h
        if aspect < 0.7 and aspect > 0.2 and area > 100 and area < 3000:
            filtered_coordinates = (x, y, w, h)
            filtered_rectangles.append(filtered_coordinates)
            mode_samples.append(y)
    filtered_rectangles = list(set(filtered_rectangles))
    if len(mode_samples) == 0:
        return 'Error'
    else:
        mode = stat.mode(mode_samples)
        min_distance = min_euclidean(filtered_rectangles)
        for i in range(len(filtered_rectangles)):
            for j in range(i+1, len(filtered_rectangles)):
                [x1, y1, w1, h1] = [filtered_rectangles[i][0], filtered_rectangles[i]
                                    [1], filtered_rectangles[i][2], filtered_rectangles[i][3]]
                [x2, y2, w2, h2] = [filtered_rectangles[j][0], filtered_rectangles[j]
                                    [1], filtered_rectangles[j][2], filtered_rectangles[j][3]]
                u = (x1, y1)
                v = (x2, y2)
                euclid = distance.euclidean(u, v)
                if euclid < min_distance * 3:
                    a = (x1, y1, w1, h1)
                    b = (x2, y2, w2, h2)
                    posible_characters.append(a)
                    posible_characters.append(b)
        posible_characters = list(set(posible_characters))
        posible_characters = sorted(
            posible_characters, key=operator.itemgetter(0), reverse=False)
        for i in range(len(posible_characters)):
            res = posible_characters[i][1]-mode
            if abs(res) < mode_error:
                characters.append(posible_characters[i])
        if len(characters) == 0:
            return 'Error'
        for i in range(len(characters)):
            x, y, w, h = characters[i][0], characters[i][1], characters[i][2], characters[i][3]
            cropped = th3[y:y+h, x-int(w/10):x+w+int(w/12)]
            ret, inv_cropped = cv2.threshold(
                cropped, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            if str(type(inv_cropped)) == "<class 'NoneType'>":
                text = ''
            else:
                text = pytesseract.image_to_string(
                    inv_cropped, config='--psm 10')
                if len(text) > 1:
                    text = text[0]
                grouped_text += (text.strip())
        numbers_list = []
        bus_number = Cleaner(grouped_text)
        bus_number = Checker(bus_number)
        return bus_number


def quickAnalize(data):
    characters = []
    if len(data) == 0:
        return 'No bus read'
    print(data)
    for i in range(len(data)):
        characters.append(list(data[i]))
    mode = stat.mode(characters[:])
    print(mode)


def AnalizeCharacters(data):
    characters = []
    filtered_characters = []
    final_string = ''
    for i in range(len(data)):
        characters.append(list(data[i]))
    total = 0
    maxLen = []
    for i in data:
        total += len(i)
        maxLen.append(len(i))
    ave_size = int(int(total) / int(len(data)))
    max_size = max(maxLen)
    for i in range(max_size):
        for j in range(len(characters)):
            try:
                filtered_characters.append(characters[j][i])
            except IndexError:
                continue
        if len(filtered_characters) == 0:
            continue
        i = 0
        final_string += stat.mode(filtered_characters)
        filtered_characters = []
    return final_string

# THESE METHODS WERE USED TO GET GRAPHS FROM THE BEHAVIOUR OF THE ALGORITHM, THEY DO NOT AFFECT IT AT ALL


def countSuccess(data, busNumber):
    print('Counting... ' + busNumber)
    if len(data) == 0:
        return 'No bus read'
    success = 0
    failure = 0
    partial = 0
    partialcount = []
    errorCount = []
    successCount = []
    for i in range(0, len(data)):
        try:
            if data[i] == busNumber:
                success += 1
            if data[i] != 'Error' and data[i] != '' and data[i] != busNumber:
                partial += 1
            if data[i] == 'Error':
                failure += 1
            successCount.append(success)
            errorCount.append(failure)
            partialcount.append(partial)
        except IndexError:
            continue
    return successCount, errorCount, partialcount


def graphCount(successResult, partialResult, failureResult):
    plt.title('Success, failure and partial readings', fontsize=16)
    names = ['Success', 'Failure', 'Partial']
    values = [successResult.pop(), failureResult.pop(), partialResult.pop()]
    plt.bar(names, values)
    plt.ylabel("Count", fontsize=20)
    plt.show()


def getReadingPercentage(successResult, partialResult, failureResult):
    successPercent = []
    partialPercent = []
    failurePercent = []
    for i in range(1, 175):
        try:
            pSuccess = successResult[i-1] / i
            pPartial = partialResult[i-1] / i
            pFailure = failureResult[i-1] / i
            successPercent.append(pSuccess*100)
            partialPercent.append(pPartial*100)
            failurePercent.append(pFailure*100)
        except IndexError:
            continue
    return successPercent, partialPercent, failurePercent


def graphResults(successPercent, partialPercent, failurePercent):
    X1 = list(range(1, len(successPercent)+1, 1))
    X2 = list(range(1, len(partialPercent)+1, 1))
    X3 = list(range(1, len(failurePercent)+1, 1))
    plt.title('Success and partial %', fontsize=16)
    plt.plot(X1[0:], successPercent, 'g')  # Aciertos en verde
    plt.plot(X2[0:], partialPercent, 'b')  # Parciales en azul
    plt.xlabel("Frames", fontsize=16)
    plt.ylabel("Percentage", fontsize=16)
    plt.show()
    plt.title('Success %')
    plt.plot(X1[0:], successPercent, 'g')  # Aciertos en verde
    plt.xlabel("Frames", fontsize=16)
    plt.ylabel("Success percentage", fontsize=16)
    plt.show()
    plt.title('Failure %')
    plt.plot(X2[0:], failurePercent, 'r')  # Fallos en rojo
    plt.xlabel("Frames", fontsize=16)
    plt.ylabel("Failure percentage", fontsize=16)
    plt.show()
    plt.title('Partial %')
    plt.plot(X3[0:], partialPercent, 'b')  # Parciales en azul
    plt.xlabel("Frames", fontsize=16)
    plt.ylabel("Partial percentage", fontsize=16)
    plt.show()


def getMeanFromArray(successCount1, errorCount1, partialcount1, successCount2, errorCount2, partialcount2):
    successResult = []
    partialResult = []
    failureResult = []
    for i in range(0, 120):
        try:
            successResult.append((successCount1[i]+successCount2[i])/2)
            partialResult.append((partialcount1[i]+partialcount2[i])/2)
            failureResult.append((errorCount1[i]+errorCount2[i])/2)
        except IndexError:
            if len(successCount1) > len(successCount2):
                successResult.append(successCount1[i])
                partialResult.append(partialcount1[i])
                failureResult.append(errorCount1[i])
            else:
                successResult.append(successCount2[i])
                partialResult.append(partialcount2[i])
                failureResult.append(errorCount2[i])
    return successResult, partialResult, failureResult


"*****************MAIN*****************"
# VIDEO EXAMPLES INCLUDED IN THIS REPOSITORY
PATH = 'BUS SAMPLES/'
B022 = PATH + 'B022.mp4'
B035 = PATH + 'B035.mp4'
B107 = PATH + 'B107.mp4'
B147 = PATH + 'B147.mp4'
B148 = PATH + 'B148.mp4'
D001 = PATH + 'D001.mp4'
D037 = PATH + 'D037.mp4'
K1438 = PATH + 'K1438.mp4'
K1440 = PATH + 'K1440.mp4'
K1459 = PATH + 'K1459.mp4'


numbers = []
# EXAMPLE OF MAIN METHOD CALL FOR DIFFERENT VIDEOS
# USE INDIVIDUAL BUS VIDEOS FOR DISPLAYING SPECIFIC FRAMES FROM A BUS OR GETTING GRAPHS AND METRICS
VideoRead(B022)
VideoRead(B035)
VideoRead(B107)
VideoRead(B147)
VideoRead(B147)
VideoRead(B148)
VideoRead(D001)
VideoRead(D037)
VideoRead(K1438)
VideoRead(K1440)
VideoRead(K1459)
