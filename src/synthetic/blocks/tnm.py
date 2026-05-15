"""Блок 2: TNM-стадирование.

Зависит от: diagnosis_id, diagnosis_category, subtype_name.
Устанавливает: t_code, n_code, m_code, tumor_size_mm, n_stations, m_sites,
stage, stage_group, invasion_t_extension.
"""

import random
from core.state import ClinicalCase, weighted_choice, maybe


class TNMBlock:
    INVASION_BY_T = {
        "T2a": ["опухоль прорастает в висцеральную плевру"],
        "T3": [
            "опухоль прорастает в париетальную плевру/грудную стенку",
            "опухоль вовлекает диафрагмальный нерв",
        ],
        "T4": [
            "опухоль прорастает в средостение",
            "опухоль вовлекает крупные сосуды",
            "опухоль прорастает в тело позвонка",
        ],
    }

    def __init__(self, schemas: dict):
        self.tnm = schemas["tnm"]

    def fill(self, case: ClinicalCase, rng: random.Random) -> None:
        self._pick_t(case, rng)
        self._pick_n(case, rng)
        self._pick_m(case, rng)
        case.stage = self._lookup_stage(case.t_code, case.n_code, case.m_code)
        case.stage_group = self._stage_to_group(case.stage)

    # --- T ---

    def _pick_t(self, case: ClinicalCase, rng: random.Random) -> None:
        # AIS → Tis, MIA → T1mi
        if case.subtype_name and "in situ" in case.subtype_name.lower():
            case.t_code = "Tis"
        elif case.subtype_name and "MIA" in case.subtype_name:
            case.t_code = "T1mi"
        else:
            t_dist = self._t_distribution_for(case)
            case.t_code = weighted_choice(t_dist, rng)

        # Размер опухоли в пределах T-категории
        t_data = next(t for t in self.tnm["t_categories"] if t["code"] == case.t_code)
        if t_data["size_mm"]:
            lo, hi = t_data["size_mm"]
            case.tumor_size_mm = rng.randint(lo, hi)

        # Для T2a+ — возможен признак инвазии
        if case.t_code in self.INVASION_BY_T and maybe(0.35, rng):
            case.invasion_t_extension = rng.choice(self.INVASION_BY_T[case.t_code])

    def _t_distribution_for(self, case: ClinicalCase) -> dict:
        full = self.tnm["t_distribution"]
        if case.diagnosis_category == "carcinoid":
            # карциноиды чаще в ранней стадии
            return {k: v for k, v in full.items()
                    if k in ["T1a", "T1b", "T1c", "T2a", "T2b", "T3"]}
        # для остальных — исключаем Tis/T1mi (это только для подтипов AIS/MIA)
        return {k: v for k, v in full.items() if k not in ["Tis", "T1mi"]}

    # --- N ---

    def _pick_n(self, case: ClinicalCase, rng: random.Random) -> None:
        t_group = next(
            g for g, codes in self.tnm["t_groups"].items() if case.t_code in codes
        )
        case.n_code = weighted_choice(
            self.tnm["n_distribution_by_t_group"][t_group], rng
        )
        if case.n_code != "N0":
            case.n_stations = self._pick_n_stations(case.n_code, rng)

    def _pick_n_stations(self, n_code: str, rng: random.Random) -> list:
        all_stations = self.tnm["lymph_node_stations"]
        if n_code == "N1":
            candidates = [s for s in all_stations if s["station"].startswith(("10", "11"))]
            k = rng.randint(1, min(2, len(candidates)))
        elif n_code == "N2a":
            candidates = [s for s in all_stations if not s["station"].startswith(("10", "11"))]
            k = 1
        elif n_code == "N2b":
            candidates = [s for s in all_stations if not s["station"].startswith(("10", "11"))]
            k = rng.randint(2, min(3, len(candidates)))
        elif n_code == "N3":
            candidates = all_stations
            k = rng.randint(1, min(3, len(candidates)))
        else:
            return []

        picked = rng.sample(candidates, k)
        return [
            {
                "station": st["station"],
                "name": st["name"],
                "size_mm": rng.randint(10, 28),
                "suv": round(rng.uniform(4.0, 14.0), 1),
            }
            for st in picked
        ]

    # --- M ---

    def _pick_m(self, case: ClinicalCase, rng: random.Random) -> None:
        case.m_code = weighted_choice(
            self.tnm["m_distribution_by_n"][case.n_code], rng
        )
        if case.m_code != "M0":
            case.m_sites = self._pick_m_sites(case.m_code, rng)

    def _pick_m_sites(self, m_code: str, rng: random.Random) -> list:
        sites = self.tnm["metastatic_sites"]
        if m_code == "M1a":
            pool = [s for s in sites if s["site"] in ("pleura", "pericardium", "lung_contra")]
            return [rng.choice(pool)["ru"]]
        extra = [s for s in sites if s["site"] in ("brain", "bone", "liver", "adrenal")]
        if m_code == "M1b":
            return [rng.choice(extra)["ru"]]
        if m_code == "M1c1":
            return [rng.choice(extra)["ru"]]
        if m_code == "M1c2":
            n = rng.randint(2, min(3, len(extra)))
            return [s["ru"] for s in rng.sample(extra, n)]
        return []

    # --- Stage lookup ---

    def _lookup_stage(self, t: str, n: str, m: str) -> str:
        table = self.tnm["stage_table"]
        key = f"{t}_{n}_{m}"
        if key in table:
            return table[key]
        if m == "M0" and n == "N3":
            return table.get("ANY_N3_M0", "X")
        if m in ("M1a", "M1b", "M1c1", "M1c2"):
            return table.get(f"ANY_ANY_{m}", "X")
        # Fallback для Tis/T1mi с N>0 — приравниваем к T1a
        if t in ("Tis", "T1mi") and n != "N0":
            return self._lookup_stage("T1a", n, m)
        return "X"

    def _stage_to_group(self, stage: str) -> str:
        for group, stages in self.tnm["stage_groups"].items():
            if stage in stages:
                return group
        return "unknown"
