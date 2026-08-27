"""
platform_driver.py — spawna e MUOVE la piattaforma in Gazebo Garden (CrazySim),
sincronizzata con la stessa legge del target logico del controllore.

La piattaforma in Gazebo e' PURAMENTE VISIVA: il controllore continua a inseguire
TARGET_XY/TARGET_VEL nel main. Qui muoviamo il box Gazebo con gli STESSI numeri,
cosi' cio' che vedi coincide con cio' che il drone insegue.

USO A) test standalone (CrazySim gia' avviato):
    python3 platform_driver.py
  -> spawna il box a (3.5,3.5) e dopo 12 s lo fa scivolare a 0.2 m/s in x.

USO B) integrazione nel main (sincronia perfetta col drone) -- CONSIGLIATO:
    from platform_driver import spawn_platform, move_platform_to
    spawn_platform()                              # una volta, prima del loop
    ...
    move_platform_to(TARGET_XY[0], TARGET_XY[1])  # ogni tick, dopo aver aggiornato TARGET_XY

Il nome del world e il path dell'SDF sono gia' impostati: non devi toccare nulla.
"""

import os
import subprocess
import time
import numpy as np

# ---------------------------------------------------------------------------
# Config (gia' impostata per il tuo setup; cambia solo se cambi il main)
# ---------------------------------------------------------------------------
WORLD_NAME    = "crazysim_default"          # verificato con: gz topic -l
PLATFORM_NAME = "moving_platform"
POS0          = np.array([3.5, 3.5])        # = TARGET_XY iniziale del main
Z_PLATFORM    = 0.05                         # meta' spessore box + margine
V_PLAT        = np.array([0.2, 0.0])        # = TARGET_VEL[0:2] del main

# path dell'SDF risolto AUTOMATICAMENTE accanto a questo script -> nessun path da scrivere
SDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moving_platform.sdf")

_spawned = False


def spawn_platform():
    """Spawna il box nel mondo (una volta sola). Idempotente."""
    global _spawned
    if _spawned:
        return
    if not os.path.exists(SDF_PATH):
        print(f"[platform] ATTENZIONE: SDF non trovato in {SDF_PATH}")
    cmd = [
        "gz", "service", "-s", f"/world/{WORLD_NAME}/create",
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", f'sdf_filename: "{SDF_PATH}", name: "{PLATFORM_NAME}"',
    ]
    print("[platform] spawn:", " ".join(cmd))
    subprocess.run(cmd, check=False)
    _spawned = True
    time.sleep(0.3)  # dai a Gazebo il tempo di creare l'entita' prima di muoverla


def move_platform_to(x, y, z=Z_PLATFORM):
    """Sposta il box alla posa (x,y,z). Chiamala ogni tick col TARGET_XY corrente."""
    req = (f'name: "{PLATFORM_NAME}", '
           f'position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: 0, y: 0, z: 0, w: 1}}')
    cmd = [
        "gz", "service", "-s", f"/world/{WORLD_NAME}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "200",
        "--req", req,
    ]
    subprocess.run(cmd, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_standalone(duration=40.0, dt=0.05, start_moving_after=12.0):
    """Modo A: muove la piattaforma per conto suo (test rapido isolato)."""
    spawn_platform()
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t > duration:
            break
        if t < start_moving_after:
            pos = POS0.copy()
        else:
            pos = POS0 + V_PLAT * (t - start_moving_after)
        move_platform_to(pos[0], pos[1])
        time.sleep(dt)
    print("[platform] fine.")


if __name__ == "__main__":
    run_standalone()
