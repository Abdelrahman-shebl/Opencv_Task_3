# Import necessary modules
import cv2 as cv
import numpy as np

# Open the video file
video = cv.VideoCapture("source_video.mp4")

# Loop over each frame in the video
while video.isOpened():
    # Read the current frame
    ret, frame = video.read()

    # Get the shape of the frame
    height, width, _ = frame.shape

    # Define the region of interest in the frame
    roi = frame[450:680, 300:990]

    # Convert the ROI to grayscale
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # Blur the grayscale image
    blur = cv.GaussianBlur(gray, (5,5), 0)

    # Apply the Canny edge detector to the blurred image
    edges = cv.Canny(blur, 80, 200)

    # Apply the Hough Line Transform to the edge-detected image
    lines = cv.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Define a polygon in the frame
    pts = np.array([[700,450], [1100,680], [280,680], [570,450]])

    # Reshape the points for the fillPoly function
    pts = pts.reshape((-1, 1, 2))

    # Fill the polygon with green color
    cv.fillPoly(frame, [pts], (0, 255, 0))

    # Loop over each line detected in the Hough Line Transform
    for line in lines:
        # Get the parameters of the line
        rho, theta = line[0]

        # Convert the line parameters from polar to cartesian coordinates
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Define two points on the line at a distance of 1000 pixels from the original point
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Draw the line on the ROI
        cv.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the edge-detected image, the original frame, and the ROI
    cv.imshow("e", edges)
    cv.imshow("window", frame)
    cv.imshow("roi", roi)

    # Wait for the user to press a key
    key = cv.waitKey(30)

    # If the user presses the 'ESC' key, break the loop
    if key == 27:
        break

# Release the video file
video.release()
