import cv2
import numpy as np

# Function to detect vehicles in a frame
def detect_vehicles(frame, vehicle_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return vehicles

# Function to count vehicles and draw rectangles
def draw_vehicle_rectangles(frame, vehicles):
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Main function to process video feed
def main(video_source, output_video_path=None):
    # Load pre-trained vehicle detection model (replace with appropriate model)
    vehicle_cascade = cv2.CascadeClassifier('vehicle_detection.xml')
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        out = None
    
    # Variables for vehicle counting
    total_vehicles = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        vehicles = detect_vehicles(frame, vehicle_cascade)
        
        # Draw rectangles around vehicles
        draw_vehicle_rectangles(frame, vehicles)
        
        # Count total vehicles detected
        total_vehicles += len(vehicles)
        
        # Display frame with vehicle detection
        cv2.putText(frame, f'Total Vehicles: {total_vehicles}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Vehicle Detection', frame)
        
        # Write frame to output video if enabled
        if out is not None:
            out.write(frame)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release capture and close windows
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

# Replace 'campus_traffic.mp4' with your video source
if __name__ == '__main__':
    video_source = 'campus_traffic.mp4'
    output_video_path = 'output_video.avi'  # Replace with None if no output video is needed
    main(video_source, output_video_path)
