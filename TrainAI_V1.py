from asyncio.base_futures import _FINISHED
from math import fabs
import time
import cv2
import mss
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

torch.set_num_threads(4) # Beállítja, hogy a Ryzen 5 7600 mind a 6 magját használja az AI agyának számításaihoz!
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

            # --- ÚJ: A JÁTÉK LEFOTÓZÁSA ÉS FELDOLGOZÁSA ---
            # 1. Képernyőkép készítése
            img = np.array(self.sct.grab(self.monitor))
            # 2. Fekete-fehérré alakítás (színek nem kellenek a vezetéshez, csak lassítanák)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            # 3. Lekicsinyítjük 84x84 pixelre (Ez a szabványos méret az AI-oknál)
            resized = cv2.resize(gray, (84, 84))
            # 4. Kicsit átalakítjuk, hogy a hálózat megértse (Magasság, Szélesség, Színcsatorna)
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
        
        # === ÚJ MULTIMODÁLIS OBSERVATION SPACE ===
        self.observation_space = spaces.Dict({
            # A kamera képe (84x84 fekete-fehér)
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
            # A fizikai adatok (8 darab, A KOORDINÁTÁK NÉLKÜL!)
            "physics": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        })
        
        self.max_steps = 5000
        self.current_step = 0
        self.prev_speed = 0.0
        self.prev_cp = 0 

    def reset(self, seed=None, options=None):
        self.current_step = 0
        gc.collect() 

        while not action_q.empty(): # Agresszív ürítés
            try: action_q.get_nowait()
            except: pass
        action_q.put("RESET")
   
        # Kibontjuk az adatokat (kép is jön!)
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp, image_obs = state_q.get()
        
        # Csomagolás a Dict formátumba (Koordináták nem mennek a hálóba!)
        physics_obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear], dtype=np.float32)
        obs = {"image": image_obs, "physics": physics_obs}
        
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        while not action_q.empty():
            try: action_q.get_nowait()
            except: pass
        action_q.put(action)
        
        # Kiolvassuk az új adatokat
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp, image_obs = state_q.get()
        
        # Csomagolás a hálónak
        physics_obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear], dtype=np.float32)
        obs = {"image": image_obs, "physics": physics_obs}
   
        # 1. Kibontjuk a megnövelt csomagot
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp = state_q.get()
        
        # 2. Beletesszük a 11 adatot a tömbbe, amit az AI megkap
        obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z], dtype=np.float32)
        
        # --- EZ A SOR HIÁNYZOTT! Visszaadjuk az obs-t és egy üres info szótárat ---
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        while not action_q.empty():
            try: action_q.get_nowait()
            except: pass
        action_q.put(action)
        
        # 1. Kiolvassuk az új adatokat (Már 14 adat jön: 13 + a kép!)
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp, image_obs = state_q.get()
        
        # 2. Csomagolás a hálónak (Kép + 8 fizikai adat)
        physics_obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear], dtype=np.float32)
        obs = {"image": image_obs, "physics": physics_obs}
        
        # -----------------------------------------------------
        # JUTALMAZÁSI RENDSZER (REWARD)
        # -----------------------------------------------------
        
        # 1. Alapvető sebesség jutalom (CSAK HA ELŐRE MEGY!)
        if gear > 0:
            reward = speed * 0.1  # Előre haladásért kap pontot
        else:
            reward = -50.0        # Rükvercért hatalmas folyamatos büntetés jár!

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
            terminated = True # Ha falnak csapódik, érjen véget a kör!
                
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
    # A) TANÍTÁS MÓD
    # ==========================================
    print("\n--- NEURÁLIS HÁLÓ INICIALIZÁLÁSA (KAMERA + SZENZOROK) ---")
    # MlpPolicy helyett MultiInputPolicy!
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # --- ÚJ: Biztonsági mentés beállítása ---
    # Ez minden 100.000 lépés után csinál egy .zip fájlt a "models" nevű mappába!
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path='./models/',
        name_prefix='tm_ai_model'
    )
    
    
    # Beadjuk a callback-et a learn függvénynek
    model.learn(total_timesteps=50000000, callback=checkpoint_callback) 
    
    # Ha egyszer majd tényleg végez az 50 millióval, elmenti a végsőt is:
    model.save("tm_ai_model_final")
    
    # ==========================================
    # B) ÉLES TESZT MÓD (Most ki van kapcsolva a '#' jelekkel)
    # ==========================================
    #print("\n--- BETANÍTOTT MODELL BETÖLTÉSE ---")
    #model = PPO.load("tm_ai_model")
    #obs, info = env.reset()
    #while True:
     # action, _states = model.predict(obs, deterministic=True)
      #obs, reward, terminated, truncated, info = env.step(action)
      #if terminated or truncated:a
        #obs, info = env.reset()