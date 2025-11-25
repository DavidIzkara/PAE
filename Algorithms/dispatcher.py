# dispatcher.py
from __future__ import annotations

import os
import sys
import zarr
import numpy as np
from utils_zarr_corrected import vital_to_zarr, escribir_prediccion, STORE_PATH
from typing import List, Dict, Any

# Afegim l'arrel del projecte (VitalParser) al sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from algorithms.cardiac_output import CardiacOutput

CO_REQUIRED_TRACKS = CardiacOutput.REQUIRED_TRACKS


def prepare_zarr_for_algorithms(
    vital_path: str,
    zarr_path: str,
    algo_names: List[str],
    window_secs: float | None = None,
) -> None:
    """
    Calcula la unió de totes les REQUIRED_TRACKS dels algoritmes seleccionats
    i crida vital_to_zarr UNA sola vegada per exportar-les/actualitzar-les.
    """
    all_tracks: set[str] = set()

    for name in algo_names:
        info = ALGORITHMS.get(name)
        if info is None:
            print(f"[WARN] Algoritme desconegut: {name}")
            continue
        all_tracks.update(info["required_tracks"])

    if not all_tracks:
        print("[WARN] No hi ha tracks a exportar (cap algoritme vàlid).")
        return

    print("\n[DISPATCHER] Algoritmes seleccionats:", algo_names)
    print("[DISPATCHER] Tracks a exportar/actualitzar:")
    for t in sorted(all_tracks):
        print("   -", t)

    vital_to_zarr(
        vital_file=vital_path,
        zarr_path=zarr_path,
        tracks=sorted(all_tracks),
        window_secs=window_secs,
    )


def run_algorithms_on_zarr(zarr_path: str, algo_names: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for name in algo_names:
        info = ALGORITHMS.get(name)
        if info is None:
            print(f"[WARN] Algoritme desconegut: {name}")
            continue

        runner = info["runner"]

        try:
            algo_instance = runner(zarr_path)
            results[name] = algo_instance

            # Exemple: guardar la SÈRIE COMPLETA de CardiacOutput com a predicció
            if (
                name == "cardiac_output"
                and getattr(algo_instance, "co_ts_ms", None) is not None
                and getattr(algo_instance, "co_values", None) is not None
                and len(algo_instance.co_ts_ms) > 0
            ):
                t_arr = np.asarray([algo_instance.t_last_ms], dtype=np.int64)
                v_arr = np.asarray([algo_instance.co_last], dtype=np.float32)


                escribir_prediccion(
                    zarr_path=zarr_path,
                    pred_name="cardiac_output",  # nom de la sèrie per a aquest algoritme
                    timestamps_ms=t_arr,
                    values=v_arr,
                    modelo_info={"source": "dispatcher", "algo": "cardiac_output"},
                )

                print(
                    f"[ALG-STORE] {name}: guardada sèrie completa "
                    f"({t_arr.size} mostres) a 'cardiac_output'"
                )


        except ValueError as e:
            print(f"[WARN] No s'ha pogut calcular '{name}': {e}")
            results[name] = None

    return results



# 🔧 CONFIG: rutes per defecte RELATIVES a l'arrel del projecte
DEFAULT_VITAL_REL = "records/3/250127/n5j8vrrsb_250127_100027.vital"
DEFAULT_ZARR_REL = STORE_PATH

# Les convertim a rutes absolutes basades en PROJECT_ROOT
DEFAULT_VITAL = os.path.join(PROJECT_ROOT, DEFAULT_VITAL_REL)
DEFAULT_ZARR = os.path.join(PROJECT_ROOT, DEFAULT_ZARR_REL)



# Catàleg d'algoritmes disponibles
ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "cardiac_output": {
        "required_tracks": CO_REQUIRED_TRACKS,
        "runner": CardiacOutput,  # es cridarà com CardiacOutput(zarr_path)
    },
    # Aquí hi podràs afegir més algoritmes en el futur:
    # "ppv": {"required_tracks": PPV_REQUIRED_TRACKS, "runner": PPV},
}



if __name__ == "__main__":
    # 1) Fem servir les rutes per defecte
    vital_path = DEFAULT_VITAL
    zarr_path = DEFAULT_ZARR

   
    print(f"Fitxer .vital origen : {vital_path}")
    print(f"Fitxer .zarr sortida : {zarr_path}")

    # 2) Tria MANUAL dels algoritmes per terminal
    print("\nAlgoritmes disponibles:")
    for name in ALGORITHMS.keys():
        print(" -", name)

    raw = input("\nEscriu els algoritmes a executar (separats per comes): ")
    algo_names = [a.strip() for a in raw.split(",") if a.strip()]

    if not algo_names:
        print("[INFO] No s'ha seleccionat cap algoritme. Surto.")
        raise SystemExit(0)

    # 3) Assegurem que les tracks necessàries són al Zarr (idempotent)
    prepare_zarr_for_algorithms(vital_path, zarr_path, algo_names, window_secs=None)

    # 4) Executem els algoritmes seleccionats sobre el Zarr
    results = run_algorithms_on_zarr(zarr_path, algo_names)

print("\n========== RESULTATS ==========")
for name, instance in results.items():
    print(f"\n[{name}]")
    if instance is None:
        print("  (no disponible per aquest cas)")
    else:
        print(" ", instance)
print("================================")
