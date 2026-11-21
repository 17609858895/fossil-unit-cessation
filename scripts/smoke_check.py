"""Check event confirmation, restart revisions and interval risks on two synthetic units."""
from unittest.mock import patch

import numpy as np
import pandas as pd

import ops_lib
from fleet_intervals import prior_outcomes_at_origin
from hazard import person_period, surv_curve


def main():
    landmark = 2010 * 4 + 3
    cohort = pd.DataFrame({"uid": ["A", "B"], "lm_qi": landmark, "year": 2010, "cap": [50., 100.]})
    # A resumes in 2014 after a long silent period. B continues at the intervening dates.
    quarters = pd.DataFrame({"uid": ["A", "A", "B", "B", "B"],
                             "qi": [landmark, landmark + 16, landmark, landmark + 12, landmark + 16]})
    with patch.object(ops_lib, "_ONQ", quarters):
        early = ops_lib.outcomes(cohort, landmark + 12)
        late = ops_lib.outcomes(cohort, landmark + 16)
    np.testing.assert_array_equal(early.y1, [1., 0.])
    np.testing.assert_array_equal(late.y1, [0., 0.])
    previous = pd.DataFrame({"origin": [2010], "H": [1], "exp_gw": [0.1]})
    assert prior_outcomes_at_origin(cohort, quarters, previous, 2012, 1).empty
    for year, expected in [(2013, 0.05), (2014, 0.0)]:
        prior = prior_outcomes_at_origin(cohort, quarters, previous, year, 1)
        np.testing.assert_allclose(prior.asof_gw, [expected])

    expanded, targets, _ = person_period(np.array([[1.], [2.]]), np.array([0.25, 2.25]),
                                          np.array([1, 0]), np.array([2010, 2010]))
    np.testing.assert_array_equal(expanded[:, -1], [1, 1, 2])
    np.testing.assert_array_equal(targets, [1, 0, 0])

    class ConstantHazard:
        def predict(self, features, raw_score=True):
            return np.full(len(features), np.log(0.1 / 0.9))

    survival = surv_curve(ConstantHazard(), np.zeros((2, 1)), kmax=3)
    np.testing.assert_allclose(survival[0], [1, .9, .81, .729])
    print("Passed event timing, restart revision, capacity and hazard checks.")


if __name__ == "__main__":
    main()
