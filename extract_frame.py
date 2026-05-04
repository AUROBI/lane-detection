import cv2

cap = cv2.VideoCapture("runs/metrics/output.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames in output video:", total)

# Read sequentially to avoid keyframe issues
target = 500
for i in range(target + 1):
    ret, frame = cap.read()
    if not ret:
        print("Failed at frame", i)
        break

if ret:
    cv2.imwrite("runs/metrics/screenshot.png", frame)
    print("Saved frame", target, "shape=" + str(frame.shape))
cap.release()
