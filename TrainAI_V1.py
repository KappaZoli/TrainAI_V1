from asyncio.base_futures import _FINISHED
from math import fabs
import time
import gc
import torch
import threading
import queue
from tkinter import END
from turtle import distance
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from tminterface.client import Client
from tminterface.interface import TMInterface

# 1. POSTAFIÓKOK LÉTREHOZÁSA (Kommunikáció a játék és az AI között)
state_q = queue.Queue(maxsize=1)
action_q = queue.Queue(maxsize=1)

torch.set_num_threads(6) # Beállítja, hogy a Ryzen 5 7600 mind a 6 magját használja az AI agyának számításaihoz!
# ==========================================
# 1. RÉSZ: A JÁTÉK MOTORJA (A Test)
# ==========================================
class TMAIClient(Client):
    def __init__(self):
        super().__init__()
        # ÚJ: jelző, hogy a kocsi célba ért-e ebben a körben
        self.finished = False
        self.current_cp = 0

    def on_registered(self, iface: TMInterface):
        print("\n>>> Játék motor csatlakoztatva a postafiókhoz! <<<")

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
            
            # --- NYOMOZÓ MÓD: Kiíratjuk az összes létező változót a state-ből! ---
            if _time == 0: 
                print("\n>>> SCENE_MOBIL TITKAI: <<<")
                if hasattr(state, 'scene_mobil'):
                    print(dir(state.scene_mobil))
                print("\n>>> DYNA TITKAI: <<<")
                if hasattr(state, 'dyna'):
                    print(dir(state.dyna))
                print("==================================\n")
            
            # 1. Alap adatok
            speed = state.display_speed
            yaw, pitch, roll = state.yaw_pitch_roll
            vel_x, vel_y, vel_z = state.velocity
            
            # 2. A Gear okos keresése (hasattr = megnézi, hogy létezik-e az adott név)
            if hasattr(state, 'gear'):
                gear = state.gear
            elif hasattr(state, 'engine_gear'):
                gear = state.engine_gear
            elif hasattr(state, 'scene_mobil') and hasattr(state.scene_mobil, 'gear'):
                gear = state.scene_mobil.gear
            else:
                gear = 1.0 # Vészmegoldás, hogy ne fagyjon le a játék, amíg meg nem találjuk!

            # 3. Postafiók (8 adat + 2 jelző)
            if _time >= 0:
                if state_q.full():
                    try: state_q.get_nowait()
                    except: pass
                state_q.put((speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, self.finished, self.current_cp))
            
            # --- AKCIÓK ---
            try:
                if _time >= 0:
                    action = action_q.get(timeout=1.0) 
                else:
                    action = action_q.get_nowait()
            except queue.Empty:
                action = None
                
            # --- VÉGREHAJTÁS ---
            if isinstance(action, str) and action == "RESET":
                iface.execute_command("press delete")
                self.finished = False
                self.current_cp = 0 
            elif action is not None and _time >= 0:
                steer_val = int(action[0] * 65536)
                iface.execute_command(f"steer {steer_val}")
                
                if action[1] > 0.0: iface.execute_command("gas 1")
                else: iface.execute_command("gas 0")
                    
                if action[2] > 0.0: iface.execute_command("brake 1")
                else: iface.execute_command("brake 0")
                    
        except Exception as e:
            print(f"--- VÉGZETES HIBA A JÁTÉK SZÁLBAN: {e} ---")

# ==========================================
# 2. RÉSZ: AZ AI KÖRNYEZETE (Az Aréna)
# ==========================================
class TrackmaniaEnv(gym.Env):
   def __init__(self):
        super().__init__()
        # === ÚJ ACTION SPACE === 
        # 3 darab érték: [Kormány, Gáz, Fék], mindegyik -1.0 és 1.0 között mozoghat
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # === ÚJ OBSERVATION SPACE ===
        # 8 darab érték: [speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.max_steps = 5000
        self.current_step = 0
        self.prev_speed = 0.0
        self.prev_cp = 0 

   def reset(self, seed=None, options=None):
        self.current_step = 0
        gc.collect() 

        if action_q.full():
            try: action_q.get_nowait()
            except: pass
        action_q.put("RESET")
   
        # Olvassuk ki az új, 8+2 tagú listát!
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, finished, current_cp = state_q.get()
        
        self.prev_speed = speed
        self.prev_cp = current_cp 
        
        # A 8 elemet beletesszük egy tömbbe és ezt adjuk a hálónak
        return np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear], dtype=np.float32), {}

   def step(self, action):
        self.current_step += 1
        
        if action_q.full():
            try: action_q.get_nowait()
            except: pass
        action_q.put(action)
        
        # Kiolvassuk az új adatokat lépésenként
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, finished, current_cp = state_q.get()
        obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear], dtype=np.float32)
        
        # -----------------------------------------------------
        # INNENTŐL A JUTALMAZÁSI RENDSZERED (REWARD) MARAD UGYANAZ!
        # ... (Ide jön a sebesség jutalom, checkpoint bónusz, fal büntetés stb.) ...
        # -----------------------------------------------------
        
        reward = speed 
        
        steering_effort = abs(action[0])
        reward -= steering_effort * 0.05 
        
        if steering_effort < 0.1:
            reward += 1.0

        # === ÚJ: CHECKPOINT JUTALOM ===
        # Ha a jelenlegi CP nagyobb, mint a régi, akkor haladtunk előre!
        if current_cp > self.prev_cp:
            reward += 500.0  # Hatalmas jutalom a haladásért!
            print(f"🚀 BÓNUSZ: Új checkpoint elérve! ({current_cp})")
            self.prev_cp = current_cp

        speed_diff = speed - self.prev_speed
        self.prev_speed = speed 
        
        terminated = False
        
        if speed_diff < -15.0:
            reward -= 1000
            terminated = False
            
        if speed < 3.0 and self.current_step > 300:
            reward -= 100 
            if self.current_step % 100 == 0: 
                terminated = True
                
        if abs(roll) > 1.5: 
            reward -= 500 
            terminated = True
            
        

        if finished:
            reward += 1000
            terminated = True
            print(f">>> CÉL! Jutalom ezért a körért: {reward:.1f} (lépés: {self.current_step})")

        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        
        
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
    # A) TANÍTÁS MÓD (Ezt csinálja most!)
    # ==========================================
    print("\n--- NEURÁLIS HÁLÓ INICIALIZÁLÁSA ---")
    # Ha van már régi modell, folytathatjuk azt is, de most kezdjünk egy frisset:
    model = PPO("MlpPolicy", env, verbose=1)
    
     #Emeljük meg a lépésszámot 50.000-re, hogy legyen ideje kitapasztalni a kanyart!
    model.learn(total_timesteps=50000000) 
    model.save("tm_ai_model")
    
    # ==========================================
    # B) ÉLES TESZT MÓD (Most ki van kapcsolva a '#' jelekkel)
    # ==========================================
    #print("\n--- BETANÍTOTT MODELL BETÖLTÉSE ---")
    #model = PPO.load("tm_ai_model")
    #obs, info = env.reset()
    #while True:
     # action, _states = model.predict(obs, deterministic=True)
      #obs, reward, terminated, truncated, info = env.step(action)
      #if terminated or truncated:
        #obs, info = env.reset()