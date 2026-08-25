"""The iso-level a reanalysis extracts at, and why a binary design may not use the volume target.

The bug this pins was silent and it flattered the result. `reanalysis_boundary.py` chose its
threshold by bisecting on retained VOLUME until the extracted body carried the design's own
material. On a grey design that is right. On a binary one it is degenerate: every level in (0,1)
selects the same tets, so the bisection can only raise the volume by driving the level to ~0, which
admits VOID tets -- and the reanalysis then re-solves the body as fully solid, so those void wisps
become load-bearing bridges the design does not contain.

Measured on the real designs (24 Aug 2026), the correction is large and is NOT a common factor:

    h=0.10 Lagrange   >= 94.4 %  ->  >= 648.6 %      body was 9.6 % void
    h=0.25 Lagrange   >= 567 %   ->  >= 1914.3 %     body was 10.0 % void
    h=0.25 cover      >= 14.4 %  ->  >= 20.0 %       body was 3.8 % void

The asserts below use a SYNTHETIC design rather than those runs, so the test is fast and does not
depend on an npz that may be re-run. What it reproduces is the mechanism: a binary field whose solid
part is two blocks joined only through a near-void element.
"""

from __future__ import annotations

import numpy as np
import pytest


def _bisect_level(rho, vol, target, extract):
    """The OLD rule, transcribed. Kept here so the test can show it picks the wrong level."""
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if vol[extract(mid)].sum() > target:
            lo = mid
        else:
            hi = mid
    return lo


@pytest.fixture
def design():
    """20 solid cells, 1 near-void bridge, 79 void cells. Unit volume each, so vol == count."""
    rho = np.concatenate([np.ones(20), np.full(1, 1e-5), np.full(79, 1e-9)])
    vol = np.ones_like(rho)
    return rho, vol


class TestTheOldRuleIsDegenerateOnBinaryDesigns:
    def test_the_volume_target_cannot_be_met_without_admitting_void(self, design):
        rho, vol = design
        target = float((rho * vol).sum())          # 20.00001 -- essentially the solid volume
        extract = lambda t: rho > t                 # noqa: E731  (no connectivity: isolate the level)
        tau = _bisect_level(rho, vol, target, extract)
        kept = rho > tau
        assert tau < 0.5, f"the bisection should be driven below 0.5; it chose {tau}"
        assert kept.sum() > 20, (
            "the whole point: meeting the volume target on a binary design forces the level down "
            f"until void cells are admitted, but only {kept.sum()} cells were kept"
        )
        void_kept = int((kept & (rho < 0.5)).sum())
        assert void_kept >= 1, "at least the bridge cell must have been swept in"

    def test_at_rho_half_the_body_is_exactly_the_solid_set(self, design):
        rho, _vol = design
        kept = rho > 0.5
        assert kept.sum() == 20, f"rho>0.5 must select the solid set alone, got {kept.sum()}"
        assert not (kept & (rho < 0.5)).any(), "no void may enter at the honest level"


class TestTheLevelIsInsensitiveExactlyWhenTheDesignIsBinary:
    """The degeneracy has a signature the script can detect, and this is it."""

    def test_a_binary_design_gives_the_same_set_at_every_sane_level(self, design):
        rho, _ = design
        sets = [frozenset(np.where(rho > t)[0].tolist()) for t in (0.3, 0.5, 0.7)]
        assert len(set(sets)) == 1, "on a binary design the level must not change the set"

    def test_a_grey_design_does_not(self):
        rho = np.linspace(0.0, 1.0, 101)
        sets = [frozenset(np.where(rho > t)[0].tolist()) for t in (0.3, 0.5, 0.7)]
        assert len(set(sets)) == 3, "a grey design must be level-sensitive, or the guard is wrong"


class TestTheBinaryDetector:
    """`M_nd < BINARY_MND` is what selects the branch, so it has to separate the real cases."""

    @staticmethod
    def _m_nd(rho):
        return float((4 * rho * (1 - rho)).mean())

    def test_it_fires_on_the_measured_binary_designs(self, design):
        rho, _ = design
        assert self._m_nd(rho) < 0.01, f"M_nd {self._m_nd(rho)} should read as binary"
        # the real runs, for the record
        for measured in (0.0004, 0.0001, 0.0000):
            assert measured < 0.01

    def test_it_does_not_fire_on_the_measured_grey_designs(self):
        # h10_Lagrange 0.0686, h10_cover 0.0428 -- both must keep the volume-preserving path
        for measured in (0.0686, 0.0428):
            assert measured >= 0.01, f"{measured} must not be treated as binary"
