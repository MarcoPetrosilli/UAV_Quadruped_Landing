# drone_landing — workspace ROS 2

Nodo ROS 2 che **ospita** il tuo controllo di atterraggio esistente senza
cambiarne la logica. Il controllore (`controller_deploy.py`) e il main
(`main_alpha_landing.py`) sono **identici** a prima: stato letto da
`stateEstimate` via cflib, comando via `send_setpoint`, stessa frequenza.
Il main E' il nodo ROS: la classe `LandingNode` fa da nodo, il loop di controllo
e' un **timer ROS a 1/DT Hz** (single-threaded), e lo stato del drone arriva in
modo **asincrono** dal logger cflib (thread interno di cflib) in `self.latest_state`.
La logica di controllo dentro `tick()` e' identica al vecchio for-loop.
Piu' avanti potrai pubblicare pose/TF/immagini per RViz aggiungendo publisher
nel nodo, senza toccare il controllo.

## Struttura

```
ros2_ws/
└── src/drone_landing/
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/drone_landing
    ├── launch/landing.launch.py
    └── drone_landing/
        ├── __init__.py
        ├── main_alpha_landing.py    # IL TUO main, ORA E' IL NODO ROS (classe LandingNode)
        ├── controller_deploy.py     # IL TUO controllore (invariato)
        └── reachable_polytope.npz   # <-- COPIA QUI il tuo file (vedi sotto)
```

## Dipendenze

- **ROS 2** (Humble o simile) con `rclpy` e `ros2launch`.
- **Python** (nell'env che usi per volare): `cflib`, `cvxpy`, `numpy`, `scipy`,
  `matplotlib`. Queste NON sono pacchetti ROS: installale con pip nell'env che
  ROS usa a runtime. Se il tuo controllo gira in un conda env `drones`, vedi la
  nota "conda + ROS" in fondo.

## Prima della build: copia il polytope

Il controllore carica `reachable_polytope.npz`. Copialo nel package:

```bash
cp /percorso/al/tuo/reachable_polytope.npz \
   src/drone_landing/drone_landing/reachable_polytope.npz
```

(Se il file non c'e', il main fallira' quando prova a caricarlo — lo carichi tu.)

## Build

```bash
cd ros2_ws
colcon build --packages-select drone_landing
source install/setup.bash
```

## Esecuzione

Prerequisito: **CrazySim in esecuzione** (Gazebo + firmware SITL), URI
raggiungibile (default `udp://127.0.0.1:19850`, come nel main).

Poi:

```bash
ros2 run drone_landing landing_node
# oppure
ros2 launch drone_landing landing.launch.py
```

Fa esattamente quello che faceva `python3 main_alpha_landing.py`: sale, naviga,
hold, aggancia il glideslope, atterra. Stessa frequenza, stesso tutto.

## IMPORTANTE — dove vengono salvati CSV e plot

Il main scrive `last_run_plots/flight_*.csv` e i `.png` con **path relativi alla
working directory**. Con `ros2 run`, la working directory e' quella da cui lanci
il comando. Quindi:

- Lancia `ros2 run ...` dalla cartella dove vuoi che compaiano i risultati
  (es. da `ros2_ws/`), e li' troverai `last_run_plots/`.
- Stessa cosa per `reachable_polytope.npz`: il main lo cerca nella working
  directory se il path e' relativo. Per sicurezza il file e' anche nel package;
  se vuoi che venga trovato sempre, lancia dalla cartella che lo contiene, o
  passa un path assoluto in `HybridController(..., polytope_path=...)`.

(Non ho modificato la tua logica dei path per non toccare i file: e' solo una
questione di *da dove* lanci il comando.)

## Nota conda + ROS (se il controllo gira in un conda env)

ROS 2 e conda a volte litigano sul Python. Se il tuo stack (cflib/cvxpy) vive in
un conda env `drones`, il modo piu' semplice e':

1. `source /opt/ros/humble/setup.bash`
2. `conda activate drones`
3. assicurati che `rclpy` sia importabile nell'env attivo (o installa
   `rclpy`/usa il Python di sistema con i pacchetti pip necessari).

Se hai il classico conflitto "ROS non trova cflib / conda non trova rclpy",
la via pulita e' un venv con entrambi, oppure lanciare il nodo con il Python
che vede sia `rclpy` sia `cflib`. Dimmi quale env usi e ti do i comandi esatti.
```
