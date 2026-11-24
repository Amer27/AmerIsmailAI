"""Generate SYNTHETIC industrial sensor data for predictive-maintenance demos.

Every row is simulated in this script. No real plant, SCADA or customer data
is used. The goal is a realistic-looking multivariate time series with a
normal operating region and a slow degradation drift that ends in a fault,
so that classification and anomaly-detection models have something
meaningful to learn.
"""

import numpy as np
import pandas as pd


def generate(n_machines: int = 8, hours_per_machine: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for machine in range(n_machines):
        t = np.arange(hours_per_machine)
        # Degradation starts at a random point and grows until the fault.
        degrade_start = rng.integers(hours_per_machine // 3, hours_per_machine - 200)
        fault_at = degrade_start + rng.integers(100, 300)
        wear = np.clip((t - degrade_start) / max(fault_at - degrade_start, 1), 0, 1)

        # Synthetic sensor channels: vibration (mm/s), temperature (C),
        # pressure (bar), current (A). Baselines differ per machine and
        # drift slowly (load changes, seasons), so failure signatures must
        # be picked out of realistic background variation.
        base_vib = rng.uniform(1.5, 2.5)
        base_temp = rng.uniform(55, 65)
        load_steps = rng.choice([-0.4, 0.0, 0.4], size=hours_per_machine,
                                p=[0.02, 0.96, 0.02]).cumsum()
        temp_drift = np.clip(rng.normal(0, 0.05, hours_per_machine).cumsum(), -3, 3)
        vibration = (base_vib + load_steps + 0.6 * wear ** 2
                     + rng.normal(0, 0.35, hours_per_machine))
        temperature = (base_temp + temp_drift + 2.5 * wear
                       + 0.8 * vibration + rng.normal(0, 1.2, hours_per_machine))
        pressure = (rng.uniform(6.0, 7.0) + 0.15 * wear
                    + rng.normal(0, 0.25, hours_per_machine))
        current = 20 + 1.5 * vibration + rng.normal(0, 1.0, hours_per_machine)

        spikes = rng.random(hours_per_machine) < 0.01
        vibration += spikes * rng.normal(0, 1.5, hours_per_machine)

        label = (t >= fault_at).astype(int)  # 1 = failure recorded at this hour
        frames.append(pd.DataFrame({
            "machine_id": machine,
            "hour": t,
            "vibration": vibration,
            "temperature": temperature,
            "pressure": pressure,
            "current": current,
            "failure": label,
        }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv("synthetic_sensor_data.csv", index=False)
    print(f"Wrote synthetic_sensor_data.csv with {len(df)} rows "
          f"({df['failure'].sum()} failure hours, fully synthetic).")
