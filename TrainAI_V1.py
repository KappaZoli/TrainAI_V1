import time
import math
import gc
import queue
import threading

import cv2
import mss
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from tminterface.client import Client
from tminterface.interface import TMInterface

# 1. POSTAFIÓKOK LÉTREHOZÁSA (Kommunikáció a játék és az AI között)
state_q = queue.Queue(maxsize=1)
action_q = queue.Queue(maxsize=1)

torch.set_num_threads(6)  # Beállítja, hogy a Ryzen 5 7600 mind a 6 magját használja az AI agyának számításaihoz!


def _drain(q: queue.Queue):
    """Kiüríti a postafiókot mielőtt új adatot tennénk bele (maxsize=1, tehát legfeljebb
    1 elemet kell eldobni). Ezzel biztosítjuk, hogy a fogyasztó mindig csak a
    legfrissebb állapotot/akciót kapja, sose egy elavultat."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass


# ==========================================
# LIDAR - KONSTANSOK ÉS SUGÁRTÁBLA-GYORSÍTÓTÁR
# ==========================================
LIDAR_NUM_RAYS = 9
LIDAR_MAX_DIST = 84.0
_lidar_ray_cache = {}  # (h, w, num_rays) -> (d_values, xs, ys, in_bounds)


def _build_lidar_ray_table(h, w, num_rays, max_dist):
    """Egyszer kiszámolja mind a `num_rays` sugár (x, y) pixel-koordinátáit minden
    lehetséges távolságra 5-től max_dist-ig. Mivel az autó pozíciója (car_x, car_y)
    és a sugárszögek egy adott (h, w) képmérethez fixek, ez a tábla újrafelhasználható
    minden lépésnél ahelyett, hogy minden hívásnál újraszámolnánk (cos/sin) minden pixelre."""
    car_x, car_y = w // 2, h - 5
    d_values = np.arange(5, int(max_dist))  # ugyanaz a tartomány, mint az eredeti range(5, int(max_dist))
    angles = np.linspace(180, 0, num_rays)

    xs = np.empty((num_rays, d_values.size), dtype=np.int64)
    ys = np.empty((num_rays, d_values.size), dtype=np.int64)
    for i, angle in enumerate(angles):
        rad = math.radians(angle)
        xs[i] = (car_x + np.cos(rad) * d_values).astype(np.int64)
        ys[i] = (car_y - np.sin(rad) * d_values).astype(np.int64)

    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    # Csak azért clippeljük, hogy biztonságosan indexelhessünk a tömbökkel;
    # a képen kívülre eső pontokat az in_bounds maszk úgyis kizárja lentebb.
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    return d_values, xs, ys, in_bounds


def get_lidar_distances(gray_img, num_rays=LIDAR_NUM_RAYS):
    """Vizuális LIDAR szimulátor. Kiszámolja a falak távolságát a kép alapján.

    Ez funkcionálisan pontosan ugyanazt csinálja, mint az eredeti pixelenkénti
    raycast-hurok (leteszteltem, hogy a kimenet minden esetben megegyezik),
    csak a sugarak geometriáját gyorsítótárazza és NumPy-jal vektorizálja a
    pixel-kiolvasást a lassú, beágyazott Python `for` hurok helyett.
    """
    h, w = gray_img.shape
    key = (h, w, num_rays)
    table = _lidar_ray_cache.get(key)
    if table is None:
        table = _build_lidar_ray_table(h, w, num_rays, LIDAR_MAX_DIST)
        _lidar_ray_cache[key] = table
    d_values, xs, ys, in_bounds = table

    # 1. Éldetektálás (Megkeressük a vonalakat, útszéleket a képen)
    edges = cv2.Canny(gray_img, 50, 150)

    is_edge = np.zeros_like(in_bounds)
    is_edge[in_bounds] = edges[ys[in_bounds], xs[in_bounds]] > 0

    # Egy sugár akkor "áll meg", ha falba (élbe) ütközött, vagy kilépett a képből
    hit = is_edge | ~in_bounds
    any_hit = hit.any(axis=1)
    first_hit_idx = hit.argmax(axis=1)  # az első True index soronként (0, ha nincs találat)

    distances = np.where(any_hit, d_values[first_hit_idx], LIDAR_MAX_DIST).astype(np.float32)
    return distances / np.float32(LIDAR_MAX_DIST)


# ==========================================
# 1. RÉSZ: A JÁTÉK MOTORJA (A Test)
# ==========================================
class TMAIClient(Client):
    def __init__(self):
        super().__init__()
        self.finished = False
        self.current_cp = 0

        # --- Képernyőlopó inicializálása ---
        self.sct = mss.mss()
        # Beállítjuk, hogy a monitor bal felső sarkából vegyen fel egy 800x600-as részt.
        # (Ezt majd a játékod ablakához kell igazítani!)
        self.monitor = {"top": 30, "left": 0, "width": 800, "height": 600}

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        # Ezt a TMInterface automatikusan meghívja, amikor a kocsi
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

            # --- A JÁTÉK LEFOTÓZÁSA ÉS FELDOLGOZÁSA (GOLYÓÁLLÓ VERZIÓ) ---
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
                _drain(state_q)  # Agresszív ürítés a fagyás ellen!
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

# --- Jutalom-függvény konstansai (a régi "mágikus számok" kiemelve, hangolás céljából) ---
MIN_MOVING_SPEED = 2.0                 # ez alatt "meg sem mozdul / beragadt egy falba"
PENALTY_NOT_MOVING_FORWARD = -50.0     # ha áll, vagy tolat
FORWARD_SPEED_REWARD_SCALE = 0.1       # jutalom = sebesség * ez, ha előre halad

STEERING_PENALTY_SCALE = 0.05
STRAIGHT_STEERING_THRESHOLD = 0.1      # ez alatti kormányzás számít "egyenes haladásnak"
STRAIGHT_DRIVING_BONUS = 1.0

CHECKPOINT_BONUS = 500.0

CRASH_SPEED_DROP_THRESHOLD = -15.0     # ekkora hirtelen lassulás = ütközésgyanús
CRASH_PENALTY = 1000.0

WALL_HUG_LIDAR_THRESHOLD = 0.05        # bal/jobb szélső LIDAR távolság ez alatt = falat súrol
WALL_HUG_MIN_SPEED = 20.0              # csak akkor büntetjük, ha valóban halad
WALL_HUG_PENALTY = 10.0

ROLLOVER_ROLL_THRESHOLD = 1.5
ROLLOVER_PENALTY = 500.0

MIN_TRACK_POS_Y = 20.0                 # ez alatt "leesett a pályáról"
MIN_TRACK_POS_Z = 490.0

FINISH_BONUS = 1000.0


class TrackmaniaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 17 adat (8 fizika + 9 LIDAR lézer távolság)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.max_steps = 5000
        self.current_step = 0
        self.prev_speed = 0.0
        self.prev_cp = 0

    @staticmethod
    def _build_observation(state):
        """Közös segédfüggvény a nyers állapot -> observation átalakításhoz.

        A reset() és a step() korábban ezt a logikát (kép -> LIDAR -> obs
        összefűzés) egymástól függetlenül, duplikálva tartalmazta. A nyers
        (nem float32-re kerekített) fizikai értékeket is visszaadja, mert a
        jutalom-függvénynek ezekre van szüksége, nem a lekerekített obs-ra.
        """
        speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear, pos_x, pos_y, pos_z, finished, current_cp, image_obs = state

        # --- LIDAR FELDOLGOZÁS ---
        gray_image = image_obs[:, :, 0]  # kivesszük a felesleges dimenziót
        lidar_data = get_lidar_distances(gray_image)

        physics_obs = np.array([speed, yaw, pitch, roll, vel_x, vel_y, vel_z, gear], dtype=np.float32)
        obs = np.concatenate((physics_obs, lidar_data))

        return obs, speed, roll, gear, lidar_data, pos_y, pos_z, finished, current_cp

    def reset(self, seed=None, options=None):
        self.current_step = 0
        gc.collect()

        _drain(action_q)
        action_q.put("RESET")

        obs, *_ = self._build_observation(state_q.get())
        return obs, {}

    def step(self, action):
        self.current_step += 1

        _drain(action_q)
        action_q.put(action)

        obs, speed, roll, gear, lidar_data, pos_y, pos_z, finished, current_cp = self._build_observation(state_q.get())

        # -----------------------------------------------------
        # JUTALMAZÁSI RENDSZER (REWARD)
        # -----------------------------------------------------

        # 1. Alapvető sebesség jutalom (CSAK HA ELŐRE MEGY!)
        if speed < MIN_MOVING_SPEED:
            # Ha meg sem mozdul, vagy beragadt egy falba
            reward = PENALTY_NOT_MOVING_FORWARD
        elif gear > 0:
            # Előre haladásért kap pontot
            reward = speed * FORWARD_SPEED_REWARD_SCALE
        else:
            # Rükvercért hatalmas folyamatos büntetés jár!
            reward = PENALTY_NOT_MOVING_FORWARD

        # 2. Kormányzás büntetése/jutalmazása
        steering_effort = abs(action[0])
        reward -= steering_effort * STEERING_PENALTY_SCALE

        # Csak akkor kap egyenes-haladás bónuszt, ha előre megy!
        if steering_effort < STRAIGHT_STEERING_THRESHOLD and gear > 0:
            reward += STRAIGHT_DRIVING_BONUS

        # === 3. CHECKPOINT JUTALOM ===
        if current_cp > self.prev_cp:
            reward += CHECKPOINT_BONUS
            print(f"🚀 BÓNUSZ: Új checkpoint elérve! ({current_cp})")
            self.prev_cp = current_cp

        speed_diff = speed - self.prev_speed
        self.prev_speed = speed
        terminated = False

        # 4. Falnak csapódás
        if speed_diff < CRASH_SPEED_DROP_THRESHOLD:
            reward -= CRASH_PENALTY

        # === 5. FAL SÚROLÁS (Wall Hugging) BÜNTETÉS ===
        # Ha a bal (0.) vagy jobb (8.) oldali távolság kevesebb, mint 5%, miközben halad
        if lidar_data[0] < WALL_HUG_LIDAR_THRESHOLD or lidar_data[8] < WALL_HUG_LIDAR_THRESHOLD:
            if speed > WALL_HUG_MIN_SPEED:  # Csak akkor, ha valóban halad és súrolja a falat
                reward -= WALL_HUG_PENALTY  # Folyamatos, apró áramütések a fal érintéséért

        # 6. Borulás
        if abs(roll) > ROLLOVER_ROLL_THRESHOLD:
            reward -= ROLLOVER_PENALTY
            terminated = True

        # 7. Leesés a pályáról (Matematikailag stabil büntetés)
        if pos_y < MIN_TRACK_POS_Y:
            reward -= CRASH_PENALTY
            terminated = True

        if pos_z < MIN_TRACK_POS_Z:
            reward -= CRASH_PENALTY
            terminated = True

        if finished:
            reward += FINISH_BONUS
            terminated = True
            print(f">>> CÉL! Jutalom ezért a körért: {reward:.1f} (lépés: {self.current_step})")

        truncated = self.current_step >= self.max_steps

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
        model = PPO("MlpPolicy", env, verbose=1)  # Újra MlpPolicy!

        checkpoint_callback = CheckpointCallback(
            save_freq=100000,  # 100 ezer lépésenként csinál egy .zip fájlt
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
        except Exception as e:
            print(f"HIBA: Nem található/nem tölthető be a(z) '{model_path}' fájl! Biztosan tanultál már? ({e})")
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