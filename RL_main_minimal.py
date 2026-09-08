

import numpy as np

IDLE, RISING, HOLD, LANDING = 0, 1, 2, 3

Z_HOLD = 1.8
Z_LAND = 0.1
Y_TARGET = 2.0

STOP_DELTA_TRANSIT = 0.2
STOP_DELTA_LAND = 0.1


class DroneMission:

    def __init__(self, x0, y0, z0):
        self.WP = {
            "rising":  np.array([x0, y0, Z_HOLD]),
            "hold":    np.array([x0, Y_TARGET, Z_HOLD]),
            "landing": np.array([x0, Y_TARGET, Z_LAND]),
        }
        self.state = "rising"

    def target(self):
        return self.WP[self.state]

    def step(self, pos):

        distance = np.linalg.norm(self.target() - pos)

        if self.state == "landing" and distance <= STOP_DELTA_LAND:
            self.state = "idle"

        elif self.state == "rising" and distance <= STOP_DELTA_TRANSIT:
            self.state = "hold"

        elif self.state == "hold" and distance <= STOP_DELTA_TRANSIT:
            self.state = "landing"


# ---- due droni, posizioni di partenza speculari in x -----------------------
drones = {
    "drone_a": DroneMission(x0=-0.5, y0=0.0, z0=0.0),
    "drone_b": DroneMission(x0=+0.5, y0=0.0, z0=0.0),
}


def main():
    while True:
        for name, d in drones.items():

            # TODO: read actual drone position (state/observation)

            # TODO: RL policy computes control action toward 'target'

            # TODO: Make drone step from actual pos

        if all(d.state == "idle" for d in drones.values()):
            break


if __name__ == "__main__":
    main()