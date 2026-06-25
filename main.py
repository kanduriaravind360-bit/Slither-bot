import cv2
import mss
import numpy as np
import pyautogui
import pytesseract
import pygetwindow as gw
import time
import threading

window = gw.getWindowsWithTitle("slither.io")[0]
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pyautogui.PAUSE = 0.0
monitor_box = {
    "top": window.top,
    "left": window.left,
    "width": window.width,
    "height": window.height,
}
W = monitor_box["width"]
H = monitor_box["height"]
mask_x1 = 0
mask_y1 = H - 50
mask_x2 = 145
mask_y2 = H
length_x1 = 104
length_y1 = H - 50
length_x2 = 170
length_y2 = H - 27
capture_center_x = monitor_box["width"] // 2
capture_center_y = monitor_box["height"] // 2
def food(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    smallest_distance = float('inf')
    target_food = None
    foods = []
    lower_vibrant = np.array([0, 30, 80])
    upper_vibrant = np.array([179, 255, 255])
    food_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)
    cv2.rectangle(food_mask, (W - 220, H - 220), (W, H), 0, -1)
    cv2.rectangle(food_mask, (W - 350, 0), (W, 250), 0, -1)
    cv2.rectangle(food_mask, (mask_x1, mask_y1), (mask_x2, mask_y2), 0, -1)
    contours, hierarchy = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        if 2< M['m00'] < 400:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            area = M['m00']
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.65:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dx = cx - capture_center_x
                dy = cy - capture_center_y
                distance = np.sqrt(dx * dx + dy * dy)
                foods.append([cx, cy])
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                if distance < smallest_distance:
                    smallest_distance = distance
                    target_food = (cx, cy)
    return smallest_distance, target_food, foods

def dfood(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dfoods = []
    lower_vibrant = np.array([0, 30, 80])
    upper_vibrant = np.array([179, 255, 255])
    dfood_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)
    cv2.rectangle(dfood_mask, (W - 220, H - 220), (W, H), 0, -1)
    cv2.rectangle(dfood_mask, (W - 350, 0), (W, 250), 0, -1)
    cv2.rectangle(dfood_mask, (mask_x1, mask_y1), (mask_x2, mask_y2), 0, -1)
    contours, hierarchy = cv2.findContours(dfood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        if 400 < M['m00'] < 1500:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            area = M['m00']
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.65:
                for point in c[::10]:
                    px = point[0][0]
                    py = point[0][1]
                    dfoods.append([px, py])
                cv2.drawContours(frame, [c], -1, (255, 0, 0), 3)
    return dfoods

def threat(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(s,40,255,cv2.THRESH_BINARY_INV)
    threat_mask = cv2.bitwise_or(
        gray_mask,
        sat_mask
    )
    threats = []
    kernel = np.ones((5, 5), np.uint8)
    threat_mask = cv2.morphologyEx(threat_mask, cv2.MORPH_CLOSE, kernel)
    cv2.rectangle(threat_mask, (W - 220, H - 220), (W, H), 0, -1)
    cv2.rectangle(threat_mask, (W - 350, 0), (W, 250), 0, -1)
    cv2.rectangle(threat_mask, (mask_x1, mask_y1), (mask_x2, mask_y2), 0, -1)
    contours, hierarchy = cv2.findContours(threat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        if M['m00'] > 4000:
            inside = cv2.pointPolygonTest(c,(capture_center_x, capture_center_y),False)
            if inside >= 0:
                cv2.drawContours(frame, [c], -1, (255, 255, 255), 4)
                continue
            for point in c[::50]:
                px = point[0][0]
                py = point[0][1]
                threats.append([px, py])
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
    return threats

def wall(frame):
    wall = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_vibrant = np.array([0, 100, 100])
    upper_vibrant = np.array([5, 255, 255])
    wall_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)
    cv2.rectangle(wall_mask, (W - 220, H - 220), (W, H), 0, -1)
    cv2.rectangle(wall_mask, (W - 350, 0), (W, 250), 0, -1)
    cv2.rectangle(wall_mask, (mask_x1, mask_y1), (mask_x2, mask_y2), 0, -1)
    contours, hierarchy = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        if M['m00'] > 10000:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 5)
            for point in c[::50]:
                px = point[0][0]
                py = point[0][1]
                wall.append([px, py])
    return wall

def get_sector_scores(foods, dfoods, threats, wall):
    sector_scores = [0.0] * 64
    for food in foods:
        dx = food[0] - capture_center_x
        dy = capture_center_y - food[1]
        distance = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        angle = (angle + 2*np.pi) % (2*np.pi)
        sector = int(angle / (np.pi/32))
        score = 100 / (distance + 1)
        sector_scores[sector] += score
        sector_scores[(sector - 1) % 64] += score * 0.5
        sector_scores[(sector + 1) % 64] += score * 0.5
        sector_scores[(sector - 2) % 64] += score * 0.25
        sector_scores[(sector + 2) % 64] += score * 0.25
    for threat in threats:
        dx = threat[0] - capture_center_x
        dy = capture_center_y - threat[1]
        distance = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        angle = (angle + 2*np.pi) % (2*np.pi)
        sector = int(angle / (np.pi/32))
        score = 600 / (distance + 1)
        sector_scores[sector] -= score
        sector_scores[(sector - 1) % 64] -= score * 0.65
        sector_scores[(sector + 1) % 64] -= score * 0.65
        sector_scores[(sector - 2) % 64] -= score * 0.4
        sector_scores[(sector + 2) % 64] -= score * 0.4
    for wallp in wall:
        dx = wallp[0] - capture_center_x
        dy = capture_center_y - wallp[1]
        distance = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        angle = (angle + 2*np.pi) % (2*np.pi)
        sector = int(angle / (np.pi/32))
        score = 700 / (distance + 1)
        sector_scores[sector] -= score
        sector_scores[(sector - 1) % 64] -= score * 0.65
        sector_scores[(sector + 1) % 64] -= score * 0.65
        sector_scores[(sector - 2) % 64] -= score * 0.4
        sector_scores[(sector + 2) % 64] -= score * 0.4
    for dfood in dfoods:
        dx = dfood[0] - capture_center_x
        dy = capture_center_y - dfood[1]
        distance = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        sector = int(angle / (np.pi / 32))
        score = 300 / (distance + 1)
        sector_scores[sector] += score
        sector_scores[(sector - 1) % 64] += score * 0.45
        sector_scores[(sector + 1) % 64] += score * 0.45
        sector_scores[(sector - 2) % 64] += score * 0.25
        sector_scores[(sector + 2) % 64] += score * 0.25
    return sector_scores

current_length = 0
latest_frame = None
frame_lock = threading.Lock()
frame_count = 0
for window in gw.getAllWindows():
    if window.title == "slither.io - Brave":
        window.activate()
        break
time.sleep(1)

def length_reader():
    global current_length
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            crop = latest_frame[length_y1:length_y2, length_x1:length_x2].copy()
            cv2.imshow("Length Crop", crop)
            cv2.waitKey(1)
        text = pytesseract.image_to_string(crop)
        try:
            current_length = int(text)
            print(current_length)
        except:
            pass
        time.sleep(0.5)

ocr_thread = threading.Thread(target=length_reader, daemon=True)
ocr_thread.start()

with mss.mss() as sct:
    while True:
        raw_img = np.array(sct.grab(monitor_box))
        frame = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
        with frame_lock:
            latest_frame = frame.copy()
        dfood_data = dfood(frame)
        wall_data = wall(frame)
        food_data = food(frame)
        threat_data = threat(frame)
        smallest_distancef, target_food, foods = food_data
        threats = threat_data
        sector_scores = get_sector_scores(foods, dfood_data, threats, wall_data)
        for sector in range(64):
            angle = (sector + 0.5) * (2 * np.pi / 64)
            x = int(capture_center_x + np.cos(angle) * 300)
            y = int(capture_center_y - np.sin(angle) * 300)
            score = sector_scores[sector]
            if score > 0:
                color = (0, 255, 0)
            elif score < 0:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.putText(frame,f"{score:.1f}",(x + 5, y),cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1)
        best_sector = np.argmax(sector_scores)
        current_angle = (best_sector + 0.5) * (np.pi / 32)
        vx = np.cos(current_angle)
        vy = np.sin(current_angle)
        LOOKAHEAD = 300
        mouse_x = window.left + capture_center_x + vx * LOOKAHEAD
        mouse_y = window.top + capture_center_y - vy * LOOKAHEAD
        best_score = np.max(sector_scores)
        pyautogui.moveTo(mouse_x, mouse_y)
        end_x = int(capture_center_x + vx * 200)
        end_y = int(capture_center_y - vy * 200)
        cv2.line(frame,(capture_center_x, capture_center_y),(end_x, end_y),(255, 255, 0),4)
        cv2.imshow('Slither bot vision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break