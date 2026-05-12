"""Climber body model — segment lengths, masses, joint names.

This module is data-only. The MJCF generation lives in `builder.py` so
that the body description (here) is decoupled from the XML format.

The 4 limbs are addressed by 2-letter codes everywhere:

    LH  left hand    RH  right hand
    LF  left foot    RF  right foot

Conventions kept in lockstep with `solver/body.py` so that 2D and 3D
betas can be compared limb-for-limb.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sim3d import config as cfg

Limb = Literal["LH", "RH", "LF", "RF"]
LIMBS: tuple[Limb, ...] = ("LH", "RH", "LF", "RF")
HAND_LIMBS: tuple[Limb, ...] = ("LH", "RH")
FOOT_LIMBS: tuple[Limb, ...] = ("LF", "RF")


@dataclass(frozen=True)
class Segments:
    """All segment lengths in metres. Derived from total height +
    wingspan. Hand/foot are short rigid stubs at the end of each
    chain — they're where the equality constraint to a hold attaches."""

    height_m: float
    wingspan_m: float

    @property
    def thigh(self) -> float:
        return self.height_m * cfg.THIGH_FRAC

    @property
    def shin(self) -> float:
        return self.height_m * cfg.SHIN_FRAC

    @property
    def foot_h(self) -> float:
        return self.height_m * cfg.FOOT_HEIGHT_FRAC

    @property
    def spine(self) -> float:
        return self.height_m * cfg.PELVIS_TO_CHEST_FRAC

    @property
    def head(self) -> float:
        return self.height_m * cfg.NECK_HEAD_FRAC

    @property
    def shoulder_width(self) -> float:
        return self.height_m * cfg.SHOULDER_WIDTH_FRAC

    @property
    def pelvis_width(self) -> float:
        return self.height_m * cfg.PELVIS_WIDTH_FRAC

    @property
    def arm_total(self) -> float:
        # One arm reach = (wingspan - shoulder_width) / 2.
        return max(0.10, (self.wingspan_m - self.shoulder_width) / 2.0)

    @property
    def upper_arm(self) -> float:
        # 35 cm / 67 cm arm_total (wingspan=181, shoulder=47)
        return self.arm_total * 0.522

    @property
    def forearm(self) -> float:
        # 29 cm / 67 cm arm_total
        return self.arm_total * 0.433

    @property
    def head_radius(self) -> float:
        # From head circumference 57 cm → r = 57/(2π) = 9.1 cm.
        return self.height_m * cfg.HEAD_RADIUS_FRAC

    @property
    def standing_leg(self) -> float:
        return self.thigh + self.shin + self.foot_h


@dataclass(frozen=True)
class Masses:
    """Per-segment mass in kg, summing exactly to total_mass."""

    total_mass: float

    def _frac(self, key: str) -> float:
        return cfg.MASS_FRAC[key] * self.total_mass

    # Single-segment masses (the chest/pelvis/head are once each)
    @property
    def pelvis(self) -> float:
        # Absorbs the rounding error so the totals match.
        accounted = sum(
            cfg.MASS_FRAC[k] for k in cfg.MASS_FRAC if k != "pelvis"
        ) * self.total_mass
        # Each per-side segment counts twice except pelvis/chest/head:
        side_segments = ("upper_arm", "forearm", "hand",
                         "thigh", "shin", "foot")
        side_total = sum(
            cfg.MASS_FRAC[k] for k in side_segments
        ) * self.total_mass
        # Total accounted-for excluding pelvis = single-side mass
        # already includes one of each side segment; we owe one more
        # of each, plus chest+head which are already counted once.
        already = (
            cfg.MASS_FRAC["chest"] + cfg.MASS_FRAC["head"]
        ) * self.total_mass + side_total
        return self.total_mass - already

    @property
    def chest(self) -> float:    return self._frac("chest")
    @property
    def head(self) -> float:     return self._frac("head")
    @property
    def upper_arm(self) -> float:return self._frac("upper_arm")
    @property
    def forearm(self) -> float:  return self._frac("forearm")
    @property
    def hand(self) -> float:     return self._frac("hand")
    @property
    def thigh(self) -> float:    return self._frac("thigh")
    @property
    def shin(self) -> float:     return self._frac("shin")
    @property
    def foot(self) -> float:     return self._frac("foot")


@dataclass(frozen=True)
class ClimberProfile:
    """Everything the simulator needs to know about a specific climber.

    Construct with explicit cm values — the sim3d package internally
    works in metres, conversion happens in `Segments`."""

    height_cm: float = cfg.DEFAULT_HEIGHT_CM
    wingspan_cm: float = cfg.DEFAULT_WINGSPAN_CM
    mass_kg: float = cfg.DEFAULT_MASS_KG
    name: str = "average-climber"

    # Strength caps — used to compute attach max-force when grabbing
    # holds. Real numbers from training literature: average climber
    # can hang one-handed on a jug at body weight (≈700 N), so a 1.5×
    # safety factor lets two hands easily hold body weight.
    grip_force_n: float = 1100.0
    foot_push_force_n: float = 1500.0

    @property
    def segments(self) -> Segments:
        return Segments(
            height_m=self.height_cm / 100.0,
            wingspan_m=self.wingspan_cm / 100.0,
        )

    @property
    def masses(self) -> Masses:
        return Masses(total_mass=self.mass_kg)


# ─── Limb → site-name mapping ───────────────────────────────────────────────
# These names are referenced both in the MJCF builder (where the sites
# are declared) and in the world (where attach/release look up sites
# by name). Single source of truth lives here.
LIMB_TIP_SITE = {
    "LH": "site_lh_tip",
    "RH": "site_rh_tip",
    "LF": "site_lf_tip",
    "RF": "site_rf_tip",
}

# Fixed child bodies whose origins are colocated with the contact tip sites.
# Equality constraints attach these bodies, not hand/foot body origins, so
# the physical contact point lands on the hold while wrists/ankles can pivot.
LIMB_TIP_BODY = {
    "LH": "tip_body_lh",
    "RH": "tip_body_rh",
    "LF": "tip_body_lf",
    "RF": "tip_body_rf",
}
LIMB_MOCAP_BODY = {
    "LH": "mocap_lh",
    "RH": "mocap_rh",
    "LF": "mocap_lf",
    "RF": "mocap_rf",
}
LIMB_EQUALITY = {
    "LH": "weld_lh",
    "RH": "weld_rh",
    "LF": "weld_lf",
    "RF": "weld_rf",
}
