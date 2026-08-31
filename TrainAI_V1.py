from asyncio.base_futures import _FINISHED
from math import fabs
import time
import cv2
import mss
import math
import gc
import torch
import threading
import queue
from tkinter import END
from turtle import distance, pos
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from tminterface.client import Client
from tminterface.interface import TMInterface
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. POSTAFIÓKOK LÉTREHOZÁSA (Kommunikáció a játék és az AI között)
state_q = queue.Queue(maxsize=1)
action_q = queue.Queue(maxsize=1)


torch.set_num_threads(6) # Beállítja, hogy a Ryzen 5 7600 mind a 6 magját használja az AI agyának számításaihoz!
def get_lidar_distances(gray_img, num_rays=9):
    """Vizuális LIDAR szimulátor. Kiszámolja a falak távolságát a kép alapján."""
    h, w = gray_img.shape
    
    # 1. Éldetektálás (Megkeressük a vonalakat, útszéleket a képen)
    edges = cv2.Canny(gray_img, 50, 150)
    
    # 2. Az autó pozíciója a képen (Lent, középen)
    car_x = w // 2
    car_y = h - 5
    
    distances = []
    # 9 lézersugár 180 fokban (Balról jobbra)
    angles = np.linspace(180, 0, num_rays)
    
    for angle in angles:
        rad = math.radians(angle)
        max_dist = 84.0 # A kép mérete a maximális távolság
        dist = max_dist
        
        # Raycasting (Kilőjük a lézert pixelről pixelre)
        for d in range(5, int(max_dist)): # 5-ről indul, hogy az autót magát ne lássa falnak
            x = int(car_x + math.cos(rad) * d)
            y = int(car_y - math.sin(rad) * d)
            
            # Ha kiment a képből, az a maximum
            if x < 0 or x >= w or y < 0 or y >= h:
                dist = d
                break
                
            # Ha a lézer "falba" (élbe) ütközik az éldetektált képen
            if edges[y, x] > 0:
                dist = d
                break
                
        # Normalizáljuk 0.0 és 1.0 közé a távolságot
        distances.append(dist / max_dist)
        
    return np.array(distances, dtype=np.float32)
# ==========================================
# 1. RÉSZ: A JÁTÉK MOTORJA (A Test)
# ==========================================
class TMAIClient(Client):
    def __init__(self):
        super().__init__()
        self.finished = False
        self.current_cp = 0
        
        # --- ÚJ: Képernyőlopó inicializálása ---
        self.sct = mss.mss()
        # Beállítjuk, hogy a monitor bal felső sarkából vegyen fel egy 800x600-as részt. 
        # (Ezt majd a játékod ablakához kell igazítani!)
        self.monitor = {"top": 30, "left": 0, "width": 800, "height": 600}

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        # ÚJ: ezt a TMInterface automatikusan meghívja, amikor a kocsi
        # áthalad egy checkpointon (a célvonal is checkpointnak számít).
        # Ha a jelenlegi checkpointok száma megegyezik az összessel (target),
        # akkor a kocsi célba ért.
        print(f">>> Checkpoint: {current}/{target}")
        if current == target:
            print(">>> CÉLBA ÉRT! <<<")
            self.finished = True

    def on_run_step(self, iface: TMInterface, _time: int):
        try:
            state = iface.get_simulation_state()
            
            # 1. Alap adatok
            speed = state.display_speed
            yaw, pitch, roll = state.yaw_pitch_roll
            vel_x, vel_y, vel_z = state.velocity
            pos_x, pos_y, pos_z = state.position
            
            gear = 1.0 
            if hasattr(state, 'scene_mobil') and hasattr(state.scene_mobil, 'engine'):
                gear = float(state.scene_mobil.engine.gear)
# --- ÚJ: A JÁTÉK LEFOTÓZÁSA ÉS FELDOLGOZÁSA (GOLYÓÁLLÓ VERZIÓ) ---
            try:
                # 1. Képernyőkép készítése
                img = np.array(self.sct.grab(self.monitor))
            except Exception as e:
                # Ha a Windows letiltja a képlopást (pl. letálcázod a játékot)
                print(f"Képlopási hiba (BitBlt)! Letálcáztad a játékot? Hiba: {e}")
                # Hogy ne fagyjon le a program, adunk a LIDAR-nak egy tiszta fekete képet ideiglenesen
                img = np.zeros((self.monitor["height"], self.monitor["width"], 4), dtype=np.uint8)

            # 2. Fekete-fehérré alakítás (színek nem kellenek a vezetéshez)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            # 3. Lekicsinyítjük 84x84 pixelre
            resized = cv2.resize(gray, (84, 84))
            # 4. Kicsit átalakítjuk a formátumot
            image_obs = np.expand_dims(resized, axis=-1)

            # 3. Postafiók küldése (Beletesszük a képet is!)
            if _time >= 0:
                while not state_q.empty(): # Agresszív ürítés a fagyás ellen!
                    try: state_q.get_nowait()
                    except: pass
                state_q.put((speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, self.finished, self.current_cp, image_obs))
            # --- AKCIÓK ---
            try:
                if _time >= 0:
                    action = action_q.get(timeout=1.0) 
                else:
                    action = action_q.get_nowait()
            except queue.Empty:
                action = None
                
            # --- VÉGREHAJTÁS ---
            # --- VÉGREHAJTÁS ---
            if isinstance(action, str) and action == "RESET":
                iface.execute_command("press delete")
                self.finished = False
                self.current_cp = 0 
            elif action is not None and _time >= 0:
                # 1. Kormányzás (-65536 és 65536 között)
                steer_val = int(action[0] * 65536)
                iface.execute_command(f"steer {steer_val}")
                
                # 2. GÁZ ÉS FÉK PEDÁL (Közös tengelyen a Trackmaniában!)
                if action[1] > 0.1:
                    # Gázpedál nyomva
                    iface.execute_command("gas -65536")
                elif action[2] > 0.1:
                    # Fékpedál nyomva (Negatív gáz)
                    iface.execute_command("gas 65536")
                else:
                    # Üresjárat (Nincs pedál lenyomva)
                    iface.execute_command("gas 0")
                    
        except Exception as e:
            print(f"--- VÉGZETES HIBA A JÁTÉK SZÁLBAN: {e} ---")
# ==========================================
# 2. RÉSZ: AZ AI KÖRNYEZETE (Az Aréna)
# ==========================================
class TrackmaniaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # === ÚJ: 17 adat (8 fizika + 9 LIDAR lézer távolság) ===
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        self.max_steps = 5000
        self.current_step = 0
        self.prev_speed = 0.0
        self.prev_cp = 0 

    def reset(self, seed=None, options=None):
        self.current_step = 0
        gc.collect() 

        while not action_q.empty(): 
            try: action_q.get_nowait()
            except: pass
        action_q.put("RESET")
   
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp, image_obs = state_q.get()
        
        # --- LIDAR FELDOLGOZÁS ---
        # A kép most már csak a LIDAR-nak kell, az AI már csak a távolságokat kapja!
        gray_image = image_obs[:, :, 0] # Kivesszük a felesleges dimenziót
        lidar_data = get_lidar_distances(gray_image)
        
        physics_obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear], dtype=np.float32)
        
        # Összefűzzük a 8 fizikát és a 9 lézert egyetlen 17-es tömbbé
        obs = np.concatenate((physics_obs, lidar_data))
        
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        while not action_q.empty():
            try: action_q.get_nowait()
            except: pass
        action_q.put(action)
        
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp, image_obs = state_q.get()
        
        # --- LIDAR FELDOLGOZÁS ---
        gray_image = image_obs[:, :, 0]
        lidar_data = get_lidar_distances(gray_image)
        physics_obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear], dtype=np.float32)
        obs = np.concatenate((physics_obs, lidar_data))
       
        
        # -----------------------------------------------------
        # JUTALMAZÁSI RENDSZER (REWARD)
        # -----------------------------------------------------
        
        # 1. Alapvető sebesség jutalom (CSAK HA ELŐRE MEGY!)
        if speed < 2.0: 
            # Ha meg sem mozdul, vagy beragadt egy falba
            reward = -50.0  
        elif gear > 0:
            # Előre haladásért kap pontot
            reward = speed * 0.1  
        else:
            # Rükvercért hatalmas folyamatos büntetés jár!
            reward = -50.0        

        
        # 2. Kormányzás büntetése/jutalmazása
        steering_effort = abs(action[0])
        reward -= steering_effort * 0.05 
        
        # Csak akkor kap egyenes-haladás bónuszt, ha előre megy!
        if steering_effort < 0.1 and gear > 0:
            reward += 1.0

        # === 3. CHECKPOINT JUTALOM ===
        if current_cp > self.prev_cp:
            reward += 500.0  
            print(f"🚀 BÓNUSZ: Új checkpoint elérve! ({current_cp})")
            self.prev_cp = current_cp

        speed_diff = speed - self.prev_speed
        self.prev_speed = speed 
        terminated = False
        
        # 4. Falnak csapódás
        if speed_diff < -15.0:
            reward -= 1000
           
         # === 5. FAL SÚROLÁS (Wall Hugging) BÜNTETÉS ===
        # Ha a bal (0.) vagy jobb (8.) oldali távolság kevesebb, mint 5%, miközben halad
        if lidar_data[0] < 0.05 or lidar_data[8] < 0.05:
            if speed > 20.0: # Csak akkor, ha valóban halad és súrolja a falat
                reward -= 10.0 # Folyamatos, apró áramütések a fal érintéséért 
            
        # 6. Borulás
        if abs(roll) > 1.5: 
            reward -= 500 
            terminated = True
        
        # 7. Leesés a pályáról (Matematikailag stabil büntetés)
        if pos_y < 20.0:
            reward -= 1000
            terminated = True

    

        if pos_z < 490:
            reward -= 1000
            terminated = True
            
        if finished:
            reward += 1000
            terminated = True
            print(f">>> CÉL! Jutalom ezért a körért: {reward:.1f} (lépés: {self.current_step})")

        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        
        # FIGYELEM: Most már csak az új, képpel bővített 'obs'-t adjuk vissza!
        return obs, reward, terminated, truncated, {}
# ==========================================
# 3. RÉSZ: A SZÁLAK INDÍTÁSA
# ==========================================
def run_game_server():
    server = TMInterface()
    client = TMAIClient()
    server.register(client)
    while True:
        time.sleep(1)

if __name__ == '__main__':
    # 1. Elindítjuk a játék kommunikációját a háttérben
    print("Szerver szál indítása...")
    threading.Thread(target=run_game_server, daemon=True).start()
    time.sleep(2) 
    
    # 2. Létrehozzuk az AI Környezetet
    env = TrackmaniaEnv()
    
    # ==========================================
    # 🌟 A FŐKAPCSOLÓ 🌟
    # True = Gyors tanítás (Diavetítés, 80+ FPS)
    # False = Éles Teszt (Szép, sima játékmenet, a már betanult modellel)
    # ==========================================
    TRAIN_MODE = True 

    if TRAIN_MODE:
       
        print("\n--- NEURÁLIS HÁLÓ TANÍTÁSA (LIDAR SZENZOROKKAL) ---")
        model = PPO("MlpPolicy", env, verbose=1) # Újra MlpPolicy!
        
        checkpoint_callback = CheckpointCallback(
            save_freq=100000, # 100 ezer lépésenként csinál egy .zip fájlt
            save_path='./models/',
            name_prefix='tm_ai_model'
        )
        
        print("Tanítás indul! Ha látni akarod mit tanult, állítsd a TRAIN_MODE-ot False-ra!")
        model.learn(total_timesteps=50000000, callback=checkpoint_callback) 
        model.save("tm_ai_model_final")
        
    else:
        print("\n--- ÉLES TESZT MÓD (Látványos vezetés) ---")
        
        # IDE ÍRD BE ANNAK A .ZIP FÁJLNAK A NEVÉT, AMIT BE AKARSZ TÖLTENI!
        # (Nézd meg a 'models' mappádban, mi a legutolsó mentés neve)
        model_path = "./models/tm_ai_model_800000_steps" 
        
        try:
            model = PPO.load(model_path)
            print(f"Modell betöltve: {model_path}")
        except:
            print("HIBA: Nem található a megadott .zip fájl! Biztosan tanultál már?")
            exit()
            
        obs, info = env.reset()
        
        while True:
            # deterministic=True: Az AI nem kísérletezik véletlenszerűen, 
            # hanem a lehető legjobb, legbiztosabb tudását használja!
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- A VARÁZSLAT A SZÉP KÉPÉRT ---
            # Picit megállítjuk a Pythont, hogy a játék motorjának legyen 
            # ideje renderelni egy szép, sima képkockát (~50 FPS-re lassítjuk)
            time.sleep(0.02) 
            
            if terminated or truncated:
                print(f"Kör vége! Elért jutalom az utolsó pillanatban: {reward}")
                obs, info = env.reset()