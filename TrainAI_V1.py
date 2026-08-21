from asyncio.base_futures import _FINISHED
from math import fabs
import time
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
            
            # 1. Alap adatok
            speed = state.display_speed
            yaw, pitch, roll = state.yaw_pitch_roll
            vel_x, vel_y, vel_z = state.velocity
            
            # --- ÚJ: A POZÍCIÓ KINYERÉSE ---
            pos_x, pos_y, pos_z = state.position
            
            # 2. A Sebességfokozat (Gear) 
            gear = 1.0 
            if hasattr(state, 'scene_mobil') and hasattr(state.scene_mobil, 'engine'):
                gear = float(state.scene_mobil.engine.gear)

            # 3. Postafiók küldése (Már 11 adat + 2 jelző!)
            if _time >= 0:
                if state_q.full():
                    try: state_q.get_nowait()
                    except: pass
                # Beletesszük a pos_x, pos_y, pos_z értékeket is a csomagba:
                state_q.put((speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, self.finished, self.current_cp))
            
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
                
                # 2. GÁZ PEDÁL (Határozott nyomás)
                # Ha az AI értéke nagyobb mint 0.1, padlógáz!
                if action[1] > 0.1:
                    iface.execute_command("gas -65536") # Ha esetleg ezzel sem megy, írd át "-65536"-ra!
                else:
                    iface.execute_command("gas 0")
                    
                # 3. FÉK PEDÁL
                if action[2] > 0.1:
                    iface.execute_command("brake 65536")
                else:
                    iface.execute_command("brake 0")
                    
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
        # Megnöveljük 11-re a bemenetek számát (hozzáadtuk a 3 koordinátát)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.max_steps = 10000
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
   
        # 1. Kibontjuk a megnövelt csomagot
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp = state_q.get()
        
        # 2. Beletesszük a 11 adatot a tömbbe, amit az AI megkap
        obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z], dtype=np.float32)
        
        # --- EZ A SOR HIÁNYZOTT! Visszaadjuk az obs-t és egy üres info szótárat ---
        return obs, {}

   def step(self, action):
        self.current_step += 1
        
        if action_q.full():
            try: action_q.get_nowait()
            except: pass
        action_q.put(action)
        
       # Kiolvassuk az új adatokat lépésenként (már a pos_x, pos_y, pos_z is benne van!)
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp = state_q.get()
        obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z], dtype=np.float32)
        
        # -----------------------------------------------------
        # INNENTŐL A JUTALMAZÁSI RENDSZERED (REWARD) MARAD UGYANAZ!
        # ... (Ide jön a sebesség jutalom, checkpoint bónusz, fal büntetés stb.) ...
        # -----------------------------------------------------
        
        reward = speed *0.1  # Sebesség jutalom
        
        steering_effort = abs(action[0])
        reward -= steering_effort * 0.05 
        
        if steering_effort < 0.1:
            reward += 1.0
# -----------------------------------------------------
        # JUTALMAZÁSI RENDSZER (REWARD) JAVÍTÁSA
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
            terminated = True # Javítva: Ha falnak csapódik, érjen véget a kör!
            
    
                
        # 6. Borulás
        if abs(roll) > 1.5: 
            reward -= 500 
            terminated = True
        
        # 7. Leesés a pályáról (Matematikailag stabil büntetés)
        if pos_y < 20.0:
            reward -= 1000
            terminated = True

        if  pos_z < 490:
            reward -= 1000
            terminated = True
            
        # (A vel_x és vel_y büntetést teljesen töröltük, mert azok térkép-irányok!)
        
        # -----------------------------------------------------
        # INNENTŐL JÖN A FINISHED RÉSZ (Az marad úgy, ahogy volt)
            
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
    # A) TANÍTÁS MÓD
    # ==========================================
    print("\n--- NEURÁLIS HÁLÓ INICIALIZÁLÁSA ---")
    model = PPO("MlpPolicy", env, verbose=1)
    
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
      #if terminated or truncated:
        #obs, info = env.reset()