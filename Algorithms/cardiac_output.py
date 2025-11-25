import numpy as np
from typing import List, Optional
from utils_zarr_corrected import leer_multiples_senyales



class CardiacOutput:
    BLD_TRACK = "Intellivue/VOL_BLD_STROKE"
    HR_CANDIDATES = [
        "Intellivue/ECG_HR",
        "Intellivue/ABP_HR",
        "Intellivue/HR",
    ]

    REQUIRED_TRACKS = HR_CANDIDATES + [BLD_TRACK]
    
    """
    CO(t) = StrokeVolume(t) * HeartRate(t)

    - Llegeix del Zarr amb utils_zarr_corrected.leer_multiples_senyales
    - Usa com a HR el primer track disponible amb dades vàlides.
    """

    def __init__(
        self,
        zarr_path: str,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
    ) -> None:
        self.zarr_path = zarr_path
        self.start_s = start_s
        self.end_s = end_s

        # Sortides principals
        self.co_ts_ms: np.ndarray | None = None
        self.co_values: np.ndarray | None = None
        self.co_last: float | None = None
        self.t_last_ms: int | None = None
        self.hr_track_used: str | None = None

        self._run()

    # ------------------------------------------------------------

    def _run(self) -> None:
        tracks = [self.BLD_TRACK] + self.HR_CANDIDATES

        datos = leer_multiples_senyales(
            self.zarr_path,
            tracks,
            start_s=self.start_s,
            end_s=self.end_s,
        )

        # 1) Stroke volume
        if self.BLD_TRACK not in datos:
            raise ValueError("No s'ha pogut llegir Intellivue/VOL_BLD_STROKE")

        df_bld = datos[self.BLD_TRACK]
        t_bld = df_bld["t_abs_ms"].to_numpy(dtype=np.int64)
        v_bld = df_bld["values"].to_numpy(dtype=np.float32)

        

        if t_bld.size == 0:
            raise ValueError("No hi ha mostres vàlides de VOL_BLD_STROKE")

        # 2) Triar HR
        hr_t = None
        hr_v = None
        used_name = None

        for cand in self.HR_CANDIDATES:
            df_hr = datos.get(cand)
            if df_hr is None:
                continue

            t_hr = df_hr["t_abs_ms"].to_numpy(dtype=np.int64)
            v_hr = df_hr["values"].to_numpy(dtype=np.float32)

            mask_hr = np.isfinite(v_hr)
            t_hr = t_hr[mask_hr]
            v_hr = v_hr[mask_hr]

            if t_hr.size == 0:
                continue

            hr_t = t_hr
            hr_v = v_hr
            used_name = cand
            break

        if hr_t is None:
            raise ValueError("No s'ha trobat cap senyal de HR amb dades")

        self.hr_track_used = used_name

        # 3) Finestra temporal comuna
        t_min = max(t_bld[0], hr_t[0])
        t_max = min(t_bld[-1], hr_t[-1])
        if t_min >= t_max:
            raise ValueError("No hi ha solapament temporal entre VOL_BLD_STROKE i HR")

        mask_win = (t_bld >= t_min) & (t_bld <= t_max)
        t_bld_win = t_bld[mask_win]
        v_bld_win = v_bld[mask_win]

        if t_bld_win.size == 0:
            raise ValueError(
                "Després de la finestra, no queden mostres de VOL_BLD_STROKE"
            )

        # 4) Interpolar HR a les marques temporals de BLD
        t_hr_f = hr_t.astype(np.float64)
        v_hr_f = hr_v.astype(np.float64)
        t_bld_f = t_bld_win.astype(np.float64)

        hr_interp = np.interp(t_bld_f, t_hr_f, v_hr_f).astype(np.float32)

        # 5) CO = SV * HR
        co_values = (v_bld_win.astype(np.float64) * hr_interp.astype(np.float64)).astype(
            np.float32
        )

        self.co_ts_ms = t_bld_win
        self.co_values = co_values

        if co_values.size > 0:
            self.co_last = float(co_values[-1])
            self.t_last_ms = int(t_bld_win[-1])

    def __repr__(self) -> str:
        if self.co_values is None or self.co_ts_ms is None:
            return "<CardiacOutput: sense dades>"

        return (
            f"<CardiacOutput: {self.co_values.size} mostres, "
            f"últim CO={self.co_last:.2f} "
            f"a t={self.t_last_ms} ms, "
            f"HR_track='{self.hr_track_used}'>"
        )
