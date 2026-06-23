import cv2
import mss
import numpy as np
import pyautogui
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pyautogui.PAUSE = 0.0
screen_w, screen_h = pyautogui.size()
center_x = screen_w // 2
center_y = screen_h // 2
monitor_box = {
    'top': center_y - 475,
    'left': center_x - 900,
    'width': 1800,
    'height': 950,
}
def food(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    smallest_distance = float('inf')
    target_food = None
    foods = []
    lower_vibrant = np.array([0, 60, 70])
    upper_vibrant = np.array([120, 255, 255])
    food_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)
    contours, hierarchy = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Food Mask", food_mask)
    for c in contours:
        M = cv2.moments(c)
        if 7< M['m00'] < 600:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            area = M['m00']
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.5:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dx = cx - 900
                dy = cy - 475
                distance = np.sqrt(dx * dx + dy * dy)
                foods.append([cx, cy])
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"({cx},{cy})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if distance < smallest_distance:
                    smallest_distance = distance
                    target_food = (cx, cy)
    if target_food is not None:
        cv2.line(frame, target_food, (900, 475), (0, 255, 255), 1)
    return smallest_distance, target_food


def threat(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threats = []
    dt = []
    _, threat_mask = cv2.threshold(gray,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    threat_mask = cv2.morphologyEx(threat_mask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(threat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Threat Mask", threat_mask)
    for c in contours:
        closest_distance = float('inf')
        closest_point = None
        M = cv2.moments(c)
        if M['m00'] > 2000:
            area = M['m00']
            x, y, w, h = cv2.boundingRect(c)
            cv2.putText(
                frame,
                str(int(np.mean(gray[y:y + h, x:x + w]))),
                (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            aspect_ratio = max(w, h) / min(w, h)
            rect_area = w * h
            fill_ratio = area / rect_area
            for point in c:
                px = point[0][0]
                py = point[0][1]
                dx = px - 900
                dy = py - 475
                distance = np.sqrt(dx * dx + dy * dy)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = (px, py)
            threat_x = closest_point[0]
            threat_y = closest_point[1]
            threats.append([threat_x, threat_y])
            dt.append(closest_distance)
            cv2.line(frame, (threat_x, threat_y),(900, 475), (0, 0, 255), 2)
            if closest_distance < 100:
                continue
            cv2.rectangle(frame,(x, y),(x + w, y + h),(255, 0, 0),2)
            cv2.putText(frame,f"A:{int(area)} AR:{aspect_ratio:.1f}",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
    return threats, dt

with mss.mss() as sct:
    while True:
        raw_img = np.array(sct.grab(monitor_box))
        frame = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
        threats, threats_distances= threat(frame)
        k = []
        for i in range(len(threats_distances)):
            k.append((threats[i], threats_distances[i]))
        smallest_distancef, target_food = food(frame)
        ordered = sorted(
            k,
            key=lambda x: x[1],
            reverse=False
        )

        if len(ordered) > 1 and ordered[1][1] < 100:
            x = ordered[1][0][0] - 900
            y = ordered[1][0][1] - 475
            xy = np.sqrt(x * x + y * y)
            if xy == 0:
                continue
            vx = (x / xy)
            vy = (y / xy)
            LOOKAHEAD = 300
            mouse_x = center_x - vx * LOOKAHEAD
            mouse_y = center_y - vy * LOOKAHEAD
            pyautogui.moveTo(mouse_x, mouse_y)
        else:
            if target_food == None:
                continue
            x = target_food[0] - 900
            y = target_food[1] - 475
            xy = np.sqrt(x * x + y * y)
            if xy == 0:
                continue
            vx = (x / xy)
            vy = (y / xy)
            LOOKAHEAD = 300
            mouse_x = center_x + vx * LOOKAHEAD
            mouse_y = center_y + vy * LOOKAHEAD
            pyautogui.moveTo(mouse_x, mouse_y)
        cv2.imshow('Slither bot vision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

