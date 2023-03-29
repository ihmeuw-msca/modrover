.. image:: https://github.com/ihmeuw-msca/modrover/workflows/build/badge.svg
    :target: https://github.com/ihmeuw-msca/modrover/actions

.. image:: https://badge.fury.io/py/modrover.svg
    :target: https://badge.fury.io/py/modrover

ModRover: Model space exploration
=================================

Simple Example
--------------

.. code-block:: python

    from modrover.rover import Rover

    # load data
    data = ...

    rover = Rover(
        # model type / family
        model_type="gaussian",
        # column name corresponding to the observation
        obs="obs",
        # covariate(s) that will be included in every model
        cov_fixed=["intercept"],
        # covariates that we want to explore
        cov_exploring=["cov_0", "cov_1"],
        # out-of-sample holdouts
        holdouts=["holdout_0", "holdout_1"],
    )

    rover.fit(data=data, strategies=["forward", "backward"], top_pct_learner=0.1)
    pred = rover.predict(data)

    # final coefficients and variance covariance matrix
    rover.super_learner.coef
    rover.super_learner.vcov

