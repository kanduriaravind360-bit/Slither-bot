import gymnasium as gym
from gymnasium import spaces
import cv2
import dxcam
import numpy as np
import pyautogui
import pygetwindow as gw
import time



pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False



class slitherenv(gym.Env):
    def __init__(self):
        super().__init__()
        self.window = gw.getWindowsWithTitle("slither.io")[0]
        self.camera = dxcam.create()
        self.camera.start(
            region=(
                self.window.left,
                self.window.top,
                self.window.right,
                self.window.bottom,
            )
        )
        self.observation_space = spaces.Box(
            low = -1000,
            high = 1000,
            shape = (256,),
            dtype = np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.play_button = cv2.imread("templatesplayagain3.png")
        self.step_count = 0
        self.last_time = time.perf_counter()
        self.current_angle = np.random.uniform(0, 2 * np.pi)
        H = self.window.bottom - self.window.top
        self.mask_x1 = 0
        self.mask_y1 = H - 50
        self.mask_x2 = 145
        self.mask_y2 = H



    def food(self, frame, hsv):
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        min_distance = float("inf")
        foods = []
        distances = []
        lower_vibrant = np.array([0, 30, 80])
        upper_vibrant = np.array([179, 255, 255])
        food_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)
        cv2.rectangle(food_mask, (width - 220, height - 220), (width, height), 0, -1)
        cv2.rectangle(food_mask, (width - 350, 0), (width, 250), 0, -1)
        cv2.rectangle(food_mask, (self.mask_x1, self.mask_y1), (self.mask_x2, self.mask_y2), 0, -1)
        contours, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if 2 < M['m00'] < 400:
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0: continue
                area = M['m00']
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity > 0.65:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    distance = np.sqrt(
                        (cx - center_x) ** 2 +
                        (cy - center_y) ** 2
                    )
                    if distance < min_distance:
                        min_distance = distance
                    distances.append(distance)
                    foods.append([cx, cy])
        if min_distance == float("inf"):
            min_distance = None
        return foods, min_distance, center_x, center_y



    def food_score(self, foods, center_x, center_y):
        sector_scores = np.zeros(64)
        for food in foods:
            dx = food[0] - center_x
            dy = center_y - food[1]
            distance = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dy, dx)
            angle = (angle + 2 * np.pi) % (2 * np.pi)
            sector = int(angle / (np.pi / 32))
            score = 100 / (distance + 1)
            sector_scores[sector] += score
        return sector_scores



    def dfood(self, frame, hsv):
        height, width = frame.shape[:2]
        dfoods = []
        lower_vibrant = np.array([0, 30, 80])
        upper_vibrant = np.array([179, 255, 255])
        dfood_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)
        cv2.rectangle(dfood_mask, (width - 220, height - 220), (width, height), 0, -1)
        cv2.rectangle(dfood_mask, (width - 350, 0), (width, 250), 0, -1)
        cv2.rectangle(dfood_mask, (self.mask_x1, self.mask_y1), (self.mask_x2, self.mask_y2), 0, -1)
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
        return dfoods



    def dfood_score(self, dfoods, center_x, center_y):
        sector_scores = np.zeros(64)
        for dfood in dfoods:
            dx = dfood[0] - center_x
            dy = center_y - dfood[1]
            distance = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dy, dx)
            angle = (angle + 2 * np.pi) % (2 * np.pi)
            sector = int(angle / (np.pi / 32))
            score = 300 / (distance + 1)
            sector_scores[sector] += score
        return sector_scores

    def threat(self, frame, hsv, gray):
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        _, gray_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        s = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY_INV)
        threat_mask = cv2.bitwise_or(
            gray_mask,
            sat_mask
        )
        threats = []
        kernel = np.ones((5, 5), np.uint8)
        threat_mask = cv2.morphologyEx(threat_mask, cv2.MORPH_CLOSE, kernel)
        cv2.rectangle(threat_mask, (width - 220, height - 220), (width, height), 0, -1)
        cv2.rectangle(threat_mask, (width - 350, 0), (width, 250), 0, -1)
        cv2.rectangle(threat_mask, (self.mask_x1, self.mask_y1), (self.mask_x2, self.mask_y2), 0, -1)
        contours, hierarchy = cv2.findContours(threat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] > 4000:
                inside = cv2.pointPolygonTest(c, (center_x, center_y), False)
                if inside >= 0:
                    continue
                for point in c[::50]:
                    px = point[0][0]
                    py = point[0][1]
                    threats.append([px, py])
        return threats



    def threat_score(self, threats, center_x, center_y):
        sector_scores = np.zeros(64)
        for threat in threats:
            dx = threat[0] - center_x
            dy = center_y - threat[1]
            distance = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dy, dx)
            angle = (angle + 2 * np.pi) % (2 * np.pi)
            sector = int(angle / (np.pi / 32))
            score = 600 / (distance + 1)
            sector_scores[sector] -= score
            sector_scores[(sector - 1) % 64] -= score * 0.65
            sector_scores[(sector + 1) % 64] -= score * 0.65
            sector_scores[(sector - 2) % 64] -= score * 0.4
            sector_scores[(sector + 2) % 64] -= score * 0.4
        return sector_scores



    def wall(self,frame, hsv):
        wall = []
        height, width = frame.shape[:2]
        lower_vibrant = np.array([0, 100, 100])
        upper_vibrant = np.array([5, 255, 255])
        wall_mask = cv2.inRange(hsv, lower_vibrant, upper_vibrant)
        cv2.rectangle(wall_mask, (width - 220, height - 220), (width, height), 0, -1)
        cv2.rectangle(wall_mask, (width - 350, 0), (width, 250), 0, -1)
        cv2.rectangle(wall_mask, (self.mask_x1, self.mask_y1), (self.mask_x2, self.mask_y2), 0, -1)
        contours, hierarchy = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] > 10000:
                for point in c[::50]:
                    px = point[0][0]
                    py = point[0][1]
                    wall.append([px, py])
        return wall



    def wall_score(self, wall, center_x, center_y):
        sector_scores = np.zeros(64)
        for wallp in wall:
            dx = wallp[0] - center_x
            dy = center_y - wallp[1]
            distance = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dy, dx)
            angle = (angle + 2 * np.pi) % (2 * np.pi)
            sector = int(angle / (np.pi / 32))
            score = 2000 / (distance + 1)
            sector_scores[sector] -= score
            sector_scores[(sector - 1) % 64] -= score * 0.65
            sector_scores[(sector + 1) % 64] -= score * 0.65
            sector_scores[(sector - 2) % 64] -= score * 0.4
            sector_scores[(sector + 2) % 64] -= score * 0.4
        return sector_scores



    def get_observation(self, raw_frame):
        hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        foods, _, center_x, center_y = self.food(raw_frame, hsv)
        dfoods = self.dfood(raw_frame, hsv)
        threats = self.threat(raw_frame, hsv, gray)
        walls = self.wall(raw_frame, hsv)
        food_scores = self.food_score(foods, center_x, center_y)
        dfood_scores = self.dfood_score(dfoods, center_x, center_y)
        threat_scores = self.threat_score(threats, center_x, center_y)
        wall_scores = self.wall_score(walls, center_x, center_y)
        obs = np.concatenate([
            food_scores,
            dfood_scores,
            threat_scores,
            wall_scores,
        ]).astype(np.float32)
        return obs



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_frame = None
        while raw_frame is None:
            raw_frame = self.camera.get_latest_frame()
        frame = cv2.resize(raw_frame, (256, 256))
        if self.is_dead(frame):
            pyautogui.click(
                self.window.left + (self.window.right - self.window.left) // 2,
                self.window.top + (self.window.bottom - self.window.top) // 2
            )
            time.sleep(1)
            raw_frame = None
            while raw_frame is None:
                raw_frame = self.camera.get_latest_frame()
        self.step_count = 0
        self.current_angle = np.random.uniform(0, 2 * np.pi)
        obs = self.get_observation(raw_frame)
        return obs ,{}



    def step(self, action):
        self.step_count += 1
        turn = np.deg2rad(6)
        if action == 0:
            self.current_angle -= 2 * turn
        elif action == 1:
            self.current_angle -= turn
        elif action == 2:
            pass
        elif action == 3:
            self.current_angle += turn
        elif action == 4:
            self.current_angle += 2 * turn
        self.current_angle %= 2 * np.pi
        vx = np.cos(self.current_angle)
        vy = np.sin(self.current_angle)
        LOOKAHEAD = 300
        mouse_x = self.window.left + (self.window.right - self.window.left)//2 + vx * LOOKAHEAD
        mouse_y = self.window.top + (self.window.bottom - self.window.top)//2 - vy * LOOKAHEAD
        pyautogui.moveTo(mouse_x, mouse_y,duration=0)
        raw_frame = None
        while raw_frame is None:
            raw_frame = self.camera.get_latest_frame()
        hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        foods, min_distance, center_x, center_y = self.food(raw_frame, hsv)
        dfoods = self.dfood(raw_frame, hsv)
        threats = self.threat(raw_frame, hsv, gray)
        walls = self.wall(raw_frame, hsv)
        frame = cv2.resize(raw_frame, (256, 256))
        snake_sector = int(
            self.current_angle /
            (2 * np.pi / 64)
        ) % 64
        reward = 0
        food_sector_scores = self.food_score(foods, center_x, center_y)
        threat_sector_scores = self.threat_score(threats, center_x, center_y)
        dfood_sector_scores = self.dfood_score(dfoods, center_x, center_y)
        wall_sector_scores = self.wall_score(walls, center_x, center_y)
        food_sector_scores = np.roll(food_sector_scores, -snake_sector)
        dfood_sector_scores = np.roll(dfood_sector_scores, -snake_sector)
        threat_sector_scores = np.roll(threat_sector_scores, -snake_sector)
        wall_sector_scores = np.roll(wall_sector_scores, -snake_sector)
        reward += (
                2.0 * food_sector_scores[0]
                + 1.6 * food_sector_scores[1]
                + 1.6 * food_sector_scores[-1]
                + 0.8 * food_sector_scores[2]
                + 0.8 * food_sector_scores[-2]
        )
        reward += (
                1.0 * threat_sector_scores[0]
                + 0.8 * threat_sector_scores[1]
                + 0.8 * threat_sector_scores[-1]
                + 0.4 * threat_sector_scores[2]
                + 0.4 * threat_sector_scores[-2]
        )
        reward += (
                1.0 * dfood_sector_scores[0]
                + 0.8 * dfood_sector_scores[1]
                + 0.8 * dfood_sector_scores[-1]
                + 0.4 * dfood_sector_scores[2]
                + 0.4 * dfood_sector_scores[-2]
        )
        reward += (
                wall_sector_scores[0]
                + 0.8 * wall_sector_scores[1]
                + 0.8 * wall_sector_scores[-1]
                + 0.4 * wall_sector_scores[2]
                + 0.4 * wall_sector_scores[-2]
        )
        reward += 0.01
        if min_distance is not None and min_distance <= 40:
            reward += 50
        done = self.is_dead(frame)
        if done:
            reward -= 500
        terminated = done
        truncated = False
        if done:
            time.sleep(1)
        obs = np.concatenate([
            food_sector_scores,
            dfood_sector_scores,
            threat_sector_scores,
            wall_sector_scores,
        ]).astype(np.float32)
        return obs, reward, terminated, truncated, {}



    def is_dead(self, frame):
        roi = frame[90:170, 60:200]
        result = cv2.matchTemplate(
            roi,
            self.play_button,
            cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > 0.95:
            return True
        else:
            return False
