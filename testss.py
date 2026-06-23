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
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    smallest_distance = float('inf')
    target_food = None
    foods = []
    lower_vibrant = np.array([0, 60, 70])
    upper_vibrant = np.array([120, 255, 255])
    food_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)
    cv2.rectangle(food_mask,(1580, 730),(1800, 950),0,-1)
    cv2.rectangle(food_mask,(1450, 0),(1800, 250),0,-1)
    contours, hierarchy = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Food Mask", food_mask)
    for c in contours:
        M = cv2.moments(c)
        if 2< M['m00'] < 1300:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            area = M['m00']
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.65:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dx = cx - 900
                dy = cy - 475
                distance = np.sqrt(dx * dx + dy * dy)
                foods.append([cx, cy])
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                if distance < smallest_distance:
                    smallest_distance = distance
                    target_food = (cx, cy)
    return smallest_distance, target_food, foods


def threat(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, gray_mask = cv2.threshold(
        gray,
        70,
        255,
        cv2.THRESH_BINARY
    )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    _, sat_mask = cv2.threshold(
        s,
        40,
        255,
        cv2.THRESH_BINARY_INV
    )

    threat_mask = cv2.bitwise_or(
        gray_mask,
        sat_mask
    )
    threats = []
    dt = []
    kernel = np.ones((5, 5), np.uint8)
    threat_mask = cv2.morphologyEx(threat_mask, cv2.MORPH_CLOSE, kernel)
    cv2.rectangle(threat_mask, (1450, 0), (1800, 250), 0, -1)
    contours, hierarchy = cv2.findContours(threat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Threat Mask", threat_mask)
    for c in contours:
        M = cv2.moments(c)
        if M['m00'] > 3000:
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
            for point in c[::10]:
                px = point[0][0]
                py = point[0][1]
                inside = cv2.pointPolygonTest(
                    c,
                    (900, 475),
                    False
                )
                if inside >= 0:
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
                    continue
                threats.append([px,py])
            cv2.rectangle(frame,(x, y),(x + w, y + h),(255, 0, 0),2)
            cv2.putText(frame,f"A:{int(area)} AR:{aspect_ratio:.1f}",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)

    return threats, dt

def get_sector_scores(foods, threats):
    sector_scores = [0.0] * 64
    for food in foods:
        dx = food[0] - 900
        dy = 475 - food[1]
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
        dx = threat[0] - 900
        dy = 475 - threat[1]
        distance = np.sqrt(dx*dx + dy*dy)
        if np.sqrt(dx * dx + dy * dy) < 150:
            continue
        angle = np.arctan2(dy, dx)
        angle = (angle + 2*np.pi) % (2*np.pi)
        sector = int(angle / (np.pi/32))
        score = 500 / (distance + 1)
        sector_scores[sector] -= score
        sector_scores[(sector - 1) % 64] -= score * 0.65
        sector_scores[(sector + 1) % 64] -= score * 0.65
        sector_scores[(sector - 2) % 64] -= score * 0.4
        sector_scores[(sector + 2) % 64] -= score * 0.4
    return sector_scores



with mss.mss() as sct:
    current_angle = 0
    while True:
        raw_img = np.array(sct.grab(monitor_box))
        frame = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
        food_data = food(frame)
        threat_data = threat(frame)
        smallest_distancef, target_food, foods = food_data
        threats, threat_distances = threat_data
        sector_scores = get_sector_scores(foods, threats)
        for sector in range(64):

            angle = (sector + 0.5) * (2 * np.pi / 64)

            x = int(900 + np.cos(angle) * 300)
            y = int(475 - np.sin(angle) * 300)

            score = sector_scores[sector]

            if score > 0:
                color = (0, 255, 0)
            elif score < 0:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            cv2.circle(frame, (x, y), 4, color, -1)

            cv2.putText(
                frame,
                f"{score:.1f}",
                (x + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1
            )
        best_sector = np.argmax(sector_scores)
        current_angle = (best_sector + 0.5) * (np.pi / 32)
        vx = np.cos(current_angle)
        vy = np.sin(current_angle)

        LOOKAHEAD = 300
        mouse_x = center_x + vx * LOOKAHEAD
        mouse_y = center_y - vy * LOOKAHEAD
        pyautogui.moveTo(mouse_x, mouse_y)
        end_x = int(
            900 + vx * 200
        )
        end_y = int(
            475 - vy * 200
        )
        cv2.line(
            frame,
            (900, 475),
            (end_x, end_y),
            (255, 255, 0),
            4
        )
        cv2.imshow('Slither bot vision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

