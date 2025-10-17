import cv2
import sys
from ultralytics import YOLO

# --- CONFIGURATION (With Dynamic Waiting Time Logic) ---

# Lane boxes are correct based on your last image.
LANE_BOXES = [
    [430, 80, 690, 240],    # Lane 0: Top Lane
    [750, 100, 950, 250],   # Lane 1: Top-Right Lane
    [500, 650, 700, 720]    # Lane 2: Bottom Lane
]

# Violation lines (stop lines) for each lane.
VIOLATION_LINE_Y_LANE0 = 250
VIOLATION_LINE_X_LANE1 = 740
VIOLATION_LINE_Y_LANE2 = 640

# --- DYNAMIC TRAFFIC LIGHT TIMING (in seconds) ---
MINIMUM_GREEN_TIME = 7      # The base time every lane gets.
MAXIMUM_GREEN_TIME = 45     # The absolute maximum green time.
SECONDS_PER_VEHICLE = 3   # The time added for each detected vehicle.
YELLOW_TIME = 3

# --- END OF CONFIGURATION ---


# --- Step 1: Load AI Model ---
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
VEHICLE_CLASSES = [2, 3, 5, 7]
MOTORCYCLE_CLASS_ID = 3
print("Model loaded successfully.")


# --- Step 2: Open Video and Get Properties ---
video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS)
YELLOW_FRAMES = int(YELLOW_TIME * fps)


# --- Step 3: Initialize State Variables ---
current_green_lane = 0
signal_timer = int(MINIMUM_GREEN_TIME * fps)
signal_state = "GREEN"
violation_ids = set()
# NEW: A list to store the predicted green time for each lane in seconds
predicted_green_times = [MINIMUM_GREEN_TIME] * len(LANE_BOXES)

print("Starting video processing...")

# --- Main Loop ---
while True:
    success, frame = cap.read()
    if not success:
        print("End of video. Exiting...")
        break

    # --- Step 4: Run AI Model with Tracking ---
    results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, verbose=False)
    
    lane_vehicle_counts = [0] * len(LANE_BOXES)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            track_id = track_ids[i]
            class_id = class_ids[i]
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Count Vehicles and check for Violations
            for j, lane_box in enumerate(LANE_BOXES):
                if lane_box[0] < center_x < lane_box[2] and lane_box[1] < center_y < lane_box[3]:
                    lane_vehicle_counts[j] += 1
                    
                    # Check for motorcycle violations
                    if class_id == MOTORCYCLE_CLASS_ID and track_id not in violation_ids:
                        violation = False
                        if j == 0 and current_green_lane != 0 and center_y > VIOLATION_LINE_Y_LANE0: violation = True
                        elif j == 1 and current_green_lane != 1 and center_x < VIOLATION_LINE_X_LANE1: violation = True
                        elif j == 2 and current_green_lane != 2 and center_y < VIOLATION_LINE_Y_LANE2: violation = True
                        
                        if violation:
                            print('\a') # BEEP!
                            violation_ids.add(track_id)
                            cv2.putText(frame, "VIOLATION!", (x1, y1 - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
                    break

    # --- Step 5: DYNAMICALLY UPDATE PREDICTED GREEN TIMES FOR ALL LANES ---
    for i in range(len(LANE_BOXES)):
        calculated_time = MINIMUM_GREEN_TIME + (lane_vehicle_counts[i] * SECONDS_PER_VEHICLE)
        predicted_green_times[i] = min(calculated_time, MAXIMUM_GREEN_TIME)

    # --- Step 6: Update Signal Logic ---
    signal_timer -= 1

    if signal_state == "GREEN" and signal_timer <= 0:
        signal_state = "YELLOW"
        signal_timer = YELLOW_FRAMES

    elif signal_state == "YELLOW" and signal_timer <= 0:
        current_green_lane = (current_green_lane + 1) % len(LANE_BOXES)
        signal_state = "GREEN"
        # Set the timer using the pre-calculated time for the new green lane
        signal_timer = int(predicted_green_times[current_green_lane] * fps)


    # --- Step 7: Draw Everything on the Frame ---
    annotated_frame = results[0].plot()

    # Draw lane boxes, counts, and violation lines
    lane_names = ["Top", "Top-Right", "Bottom"]
    for i, lane_box in enumerate(LANE_BOXES):
        cv2.rectangle(annotated_frame, (lane_box[0], lane_box[1]), (lane_box[2], lane_box[3]), (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Lane {i} ({lane_names[i]}): {lane_vehicle_counts[i]}", (lane_box[0], lane_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.line(annotated_frame, (LANE_BOXES[0][0], VIOLATION_LINE_Y_LANE0), (LANE_BOXES[0][2], VIOLATION_LINE_Y_LANE0), (0, 255, 255), 2)
    cv2.line(annotated_frame, (VIOLATION_LINE_X_LANE1, LANE_BOXES[1][1]), (VIOLATION_LINE_X_LANE1, LANE_BOXES[1][3]), (0, 255, 255), 2)
    cv2.line(annotated_frame, (LANE_BOXES[2][0], VIOLATION_LINE_Y_LANE2), (LANE_BOXES[2][2], VIOLATION_LINE_Y_LANE2), (0, 255, 255), 2)


    # --- Step 8: DRAW TRAFFIC LIGHTS AND DYNAMIC WAITING TIMES ---
    light_start_y, y_offset_per_light = 120, 150

    for i in range(len(LANE_BOXES)):
        pos_y = light_start_y + (i * y_offset_per_light)
        
        # Draw the signal lights themselves (red, yellow, green circles)
        is_green = (current_green_lane == i and signal_state == "GREEN")
        is_yellow = (current_green_lane == i and signal_state == "YELLOW")
        is_red = (current_green_lane != i)

        red_color = (0, 0, 255) if is_red else (50, 50, 50)
        yellow_color = (0, 255, 255) if is_yellow else (50, 50, 50)
        green_color = (0, 255, 0) if is_green else (50, 50, 50)
        
        cv2.putText(annotated_frame, f"Lane {i} ({lane_names[i]})", (30, pos_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(annotated_frame, (50, pos_y), 15, red_color, -1)
        cv2.circle(annotated_frame, (50, pos_y + 40), 15, yellow_color, -1)
        cv2.circle(annotated_frame, (50, pos_y + 80), 15, green_color, -1)
        
        # --- NEW: Calculate and Display Waiting Time for RED lights ---
        if is_red:
            wait_time = 0
            # Add remaining time of the current green/yellow light
            wait_time += signal_timer / fps
            
            # Iterate through the lanes that are in between this red light and the current green light
            lane_in_cycle = (current_green_lane + 1) % len(LANE_BOXES)
            while lane_in_cycle != i:
                wait_time += predicted_green_times[lane_in_cycle] + YELLOW_TIME
                lane_in_cycle = (lane_in_cycle + 1) % len(LANE_BOXES)
            
            wait_time_text = f"Wait: {int(wait_time)}s"
            cv2.putText(annotated_frame, wait_time_text, (90, pos_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # --- Master Timer for the current signal ---
    timer_text = f"Green Lane {current_green_lane}: {max(0, int(signal_timer / fps))}s"
    cv2.putText(annotated_frame, timer_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(annotated_frame, timer_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # --- Step 9: Show the Final Frame ---
    cv2.imshow("Smart Traffic System", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Clean Up ---
cap.release()
cv2.destroyAllWindows()
print("Script finished.")