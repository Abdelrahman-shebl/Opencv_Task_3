# Import the required modules
import cv2
import numpy as np

# Initialize global variables for the selected color and a boolean state
selected_color = None
state = False

# Define a callback function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    # Access the global variables
    global selected_color, state

    # If the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the color of the pixel where the mouse was clicked
        selected_color = frame[y, x]
        print("Selected Color:", selected_color)
        # Set the state to True
        state = True

# Open the first available camera
cap = cv2.VideoCapture(0)

# Create a named window to display the video
cv2.namedWindow("Video")
# Set the mouse callback function to handle mouse events in this window
cv2.setMouseCallback("Video", mouse_callback)

# Loop indefinitely
while True:
    # Read the current frame from the camera
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV color space
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # If a color has been selected by the user
    if state:
        # Convert the selected color from BGR to HSV color space
        lab_color = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        # Define the lower and upper boundaries for the color in HSV space
        lower_color = np.array([lab_color[0] - 10, lab_color[1] - 50, lab_color[2] - 50])
        upper_color = np.array([lab_color[0] + 10, lab_color[1] + 50, lab_color[2] + 50])

        # Create a mask of the pixels within the color boundaries
        mask = cv2.inRange(lab_frame, lower_color, upper_color)
        # Apply the mask to the frame to isolate the pixels within the color boundaries
        filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Detect edges in the filtered frame
        edges = cv2.Canny(filtered_frame, 50, 150)
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop over each contour
        for i, contour in enumerate(contours):
            # Skip the first contour
            if i == 0:
                continue

            # Approximate the contour to a simpler shape
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Create a copy of the filtered frame for drawing
            filtered_frame_copy = filtered_frame.copy()

            # Draw the contour on the frame copy
            cv2.drawContours(filtered_frame_copy, [contour], -1, (255, 0, 0), 2)

            # Find the bounding rectangle of the approximated contour
            x, y, w, h = cv2.boundingRect(approx)
            # Calculate the center of the bounding rectangle
            x_mid = int(x + w / 3)
            y_mid = int(y + h / 3)
            # Create a tuple of the center coordinates
            coords = (x_mid, y_mid)

            # Identify the shape of the contour based on the number of vertices
            shape = ""
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif len(approx) > 4:
                shape = "Circle"

            # Draw a label of the identified shape at the center of the bounding rectangle
            cv2.putText(filtered_frame_copy, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame copy with the drawn contours and labels
        cv2.imshow("Video", filtered_frame_copy)
    else:
        # If no color was selected, display the original frame
        cv2.imshow("Video", frame)

    # If the user presses the 'R' key
    if cv2.waitKey(1) == ord('R'):
        # Reset the selected color and the state
        selected_color = None
        state = False

    # If the user presses the 'ESC' key

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()