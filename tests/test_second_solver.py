"""planes must agree with a separate solver on small degenerate designs."""
import pytest
from experiments.solver_crosscheck import CASES,check_case


@pytest.mark.parametrize('case',range(len(CASES)))
def test_paper_planes_against_osqp(case):
    check_case(case)
