import numpy as np
import cv2 as cv
import time
num_photos=20
num_photos_taken =0
delay_pics = 7.5
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
try:
    while num_photos_taken < num_photos:    
        start=time.time()
        end = start
        while (end-start) < delay_pics:
            if (end-start) > 4.5 and (end-start) < 5.5:
                print("3...")
            elif (end-start) > 5.5 and (end-start) < 6.5:
                print("2...")
            elif(end-start) > 6.5:
                print("1...!")
            # Capture frame-by-frame
            ret, frame = cap.read()
        
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            # Display the resulting frame
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
            end = time.time()
        cv.imwrite(f'image_{num_photos_taken}.png', frame)
        print(f"Took picture {num_photos_taken}!")
        num_photos_taken += 1
    print("Done with all pictures!")
        
except KeyboardInterrupt:
    print("Exiting now")

finally:
    cap.release()
    cv.destroyAllWindows()
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()