import numpy as np
import cv2
import math

#Open Camera
capture = cv2.VideoCapture(0)
PI = math.pi

while True:
    # Capture frames from the camera
    ret, frame = capture.read()
    
    #blurred = cv2.pyrMeanShiftFiltering(frame,31,71)
    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (20, 20), (300, 300), (0, 255, 0), 0)
    crop_image = frame[20:300, 20:300]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    
    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((15, 15))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop_image,contours,0,(0,0,255),6)
    #print("number of contours : ",len(contours))
    
    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv2.convexHull(contour)

        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
        count_defects = 0
        bol = False

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)
                bol = True

            cv2.line(crop_image, start, end, [0, 255, 0], 2)
        
        
        #detect char with count_defects == 0
        
        #detect 'A' - 
        area = cv2.contourArea(contour)
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = (int(radius))
        #cv2.circle(crop_image,center,radius,(255,0,0),6)
        
        #detect B - largest area of contour
        
        #detect 'V' - 
        (x,y),(MA,ma),angle_t = cv2.fitEllipse(contour)
        #print('angle_t',angle_t)
        #print(MA,' ',ma)
        area_of_circle = PI*radius*radius
        dif = area_of_circle - area
        
        hull_test = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull_test)
        solidity = float(area)/hull_area
        
        aspect_ratio = float(w/h)
        rect_area = w*h
        print(solidity," ",aspect_ratio)
        extent = float(area)/rect_area
        
        #print(count_defects)
        #print("dif",dif)
        #print('area',area)
        
        char = '0'
        if angle_t > 100 and count_defects == 1:
            char = 'F'
        elif angle_t > 100 and count_defects == 2:
            char = 'E'
        elif count_defects == 1 and angle_t > 35 and dif > 30000:
            char = 'L'
        elif count_defects == 1 and dif < 30000 and angle_t > 24:
            char = 'C'
        elif angle_t > 27 and count_defects == 2:
            char = 'V'
        elif angle_t > 25 and area_of_circle - area < 30000 and count_defects == 0 :
            char = 'A'
        elif angle_t < 11 and area > 18000 and count_defects == 0:
            char = 'B'
        elif count_defects == 0 and angle_t < 30 and angle_t > 11:
            char = 'D'
        elif count_defects == 0 and solidity > 0.82 and aspect_ratio < 0.72:
            char = 'i'
        elif count_defects == 0:     
            char = "ONE"
        elif count_defects == 1:
            char = 'TWO'
        elif count_defects == 2:
            char = 'THREE'
        elif count_defects == 3:    
            char = "FOUR"
        elif count_defects == 4:
            char = "FIVE"
        else:
            char = 'NO CLEAR'
        #import win32com.client as wincl
        #spk = wincl.Dispatch("SAPI.SpVoice")
        cv2.putText(frame, char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)    
        #speech = "The Letter is " 
        #speech += char
        #spk.Speak(speech)
        
    except:
        pass

    # Show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

#dif 34505.924968391555
#area 15370.0
#angle_t 26.829345703125
#0.8501423763914057   0.6974358974358974
#0.848604855826474   0.6907216494845361