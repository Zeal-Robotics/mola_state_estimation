/* _
 _ __ ___   ___ | | __ _
| '_ ` _ \ / _ \| |/ _` | Modular Optimization framework for
| | | | | | (_) | | (_| | Localization and mApping (MOLA)
|_| |_| |_|\___/|_|\__,_| https://github.com/MOLAorg/mola

 Copyright (C) 2018-2026 Jose Luis Blanco, University of Almeria,
                         and individual contributors.
 SPDX-License-Identifier: GPL-3.0
 See LICENSE for full license information.
 Closed-source licenses available upon request, for this odometry package
 alone or in combination with the complete SLAM system.
*/
// Copyright 2022 Zeal Robotics

/**
 * @file   test-predict-through-odometry.cpp
 * @brief  Async backend: the {map}-frame pose of a vehicle that fuses wheel
 *         odometry is the odometry's own at the queried time, not the last
 *         solved keyframe carried forward on a velocity estimate.
 */

#include <mola_state_estimation_smoother/StateEstimationSmoother.h>
#include <mrpt/obs/CObservationOdometry.h>
#include <mrpt/poses/CPose2D.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <thread>

namespace
{
constexpr double ODOMETRY_DT  = 0.004;  // [s], 250 Hz
constexpr double ACCELERATION = 1.0;  // [m/s^2]
constexpr double ODOMETRY_X0  = 5.0;  // [m], where the odometry frame happens to start
// At rest first, so the frame of the odometry is solved from a standing vehicle.
constexpr double STANDING = 0.5;  // [s]
// The last reading falls 12 readings after one the 10 Hz decimation keeps, so the
// newest keyframe trails it by 48 ms on a vehicle doing 2 m/s.
constexpr size_t NUM_READINGS = 638;
constexpr double TOLERANCE    = 0.005;  // [m]

const char* PARAMS = R"###(
params:
    vehicle_frame_name: "base_link"
    reference_frame_name: "map"
    link_first_pose_to_reference_origin_sigma: 1e-6
    kinematic_model: KinematicModel::ConstantVelocity
    sliding_window_length: 10.0
    max_time_to_use_velocity_model: 5.0
    sigma_random_walk_acceleration_linear: 2.0
    sigma_random_walk_acceleration_angular: 1.0
    sigma_integrator_position: 0.10
    sigma_integrator_orientation: 0.10
    estimate_geo_reference: false
    odometry_min_sample_period: 0.1
    async_backend: true
)###";

double travelled(double t)
{
    const double driving = std::max(0.0, t - STANDING);
    return 0.5 * ACCELERATION * driving * driving;
}

std::optional<double> map_x(
    mola::state_estimation_smoother::StateEstimationSmoother& est, double t)
{
    const auto nav = est.estimated_navstate(mrpt::Clock::fromDouble(t), "map");
    return nav ? std::optional<double>(nav->pose.mean.x()) : std::nullopt;
}

void run_test()
{
    mola::state_estimation_smoother::StateEstimationSmoother est;
    est.initialize(mrpt::containers::yaml::FromText(PARAMS));

    double last = 0;
    for (size_t i = 0; i < NUM_READINGS; i++)
    {
        last = ODOMETRY_DT * static_cast<double>(i);
        mrpt::obs::CObservationOdometry odom;
        odom.timestamp   = mrpt::Clock::fromDouble(last);
        odom.sensorLabel = "wheels_odom";
        odom.odometry    = mrpt::poses::CPose2D(ODOMETRY_X0 + travelled(last), 0, 0);
        est.fuse_odometry(odom, "wheels_odom");
    }

    // Between the last two readings, and at the last one.
    const double between = last - 0.5 * ODOMETRY_DT;

    // The backend needs a moment to solve the frame of the odometry source.
    std::optional<double> atLast;
    std::optional<double> atBetween;
    for (int iter = 0; iter < 800; iter++)  // up to ~4 s
    {
        atLast    = map_x(est, last);
        atBetween = map_x(est, between);
        if (atLast && atBetween && std::abs(*atLast - travelled(last)) < TOLERANCE &&
            std::abs(*atBetween - travelled(between)) < TOLERANCE)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    ASSERT_(atLast.has_value());
    ASSERT_(atBetween.has_value());
    std::cout << "at the last reading: " << *atLast << " m, expected " << travelled(last) << "\n";
    std::cout << "between readings:    " << *atBetween << " m, expected " << travelled(between)
              << "\n";
    ASSERT_LT_(std::abs(*atLast - travelled(last)), TOLERANCE);
    ASSERT_LT_(std::abs(*atBetween - travelled(between)), TOLERANCE);
}
}  // namespace

int main()
{
    try
    {
        run_test();
        std::cout << "Test successful.\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
