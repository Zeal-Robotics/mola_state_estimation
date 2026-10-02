/*               _
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

/**
 * @file   FastPredictor.cpp
 * @brief  Lock-free, sub-ms pose predictor anchored on the latest backend
 *         snapshot.
 * @author Jose Luis Blanco Claraco
 */

#include "FastPredictor.h"

#include <mrpt/core/bits_math.h>  // mrpt::square
#include <mrpt/poses/CPose3DPDFGaussian.h>
#include <mrpt/poses/CPose3DPDFGaussianInf.h>
#include <mrpt/system/datetime.h>  // timeDifference

#include <mrpt/poses/Lie/SE.h>

#include <algorithm>
#include <cmath>

#include "extrapolation.h"

namespace mola::state_estimation_smoother
{

namespace
{
/// How far back the wheel odometry is kept: longer than any query lags the newest reading.
constexpr double RAW_ODOMETRY_HISTORY = 2.0;  // [s]
}  // namespace

void FastPredictor::set_snapshot(std::shared_ptr<const Snapshot> snap)
{
    std::lock_guard<std::mutex> lck(mtx_);
    snapshot_ = std::move(snap);
}

std::shared_ptr<const Snapshot> FastPredictor::snapshot() const
{
    std::lock_guard<std::mutex> lck(mtx_);
    return snapshot_;
}

void FastPredictor::note_observation_stamp(const mrpt::Clock::time_point& obsStamp)
{
    std::lock_guard<std::mutex> lck(mtx_);
    // Keep the freshest observation stamp (inputs are usually monotonic), so a
    // late arrival never drags the published stamp backwards.
    if (!lastObsStamp_ || mrpt::system::timeDifference(*lastObsStamp_, obsStamp) > 0)
    {
        lastObsStamp_     = obsStamp;
        lastObsWallclock_ = mrpt::Clock::now();
    }
}

std::optional<mrpt::Clock::time_point> FastPredictor::get_current_extrapolated_stamp() const
{
    std::lock_guard<std::mutex> lck(mtx_);
    if (!lastObsStamp_)
    {
        return std::nullopt;
    }
    return mrpt::Clock::fromDouble(
        (mrpt::Clock::nowDouble() - mrpt::Clock::toDouble(lastObsWallclock_)) +
        mrpt::Clock::toDouble(*lastObsStamp_));
}

std::optional<mrpt::Clock::time_point> FastPredictor::get_timely_stamp(
    const Parameters& params) const
{
    const auto now = get_current_extrapolated_stamp();
    std::lock_guard<std::mutex> lck(mtx_);
    if (!now || rawOdometry_.empty())
    {
        return now;
    }
    const double ahead = mrpt::system::timeDifference(rawOdometry_.back().stamp, *now);
    if (ahead <= 0 || ahead > params.max_time_to_use_velocity_model)
    {
        return now;
    }
    return rawOdometry_.back().stamp;
}

void FastPredictor::note_raw_odometry(
    const std::string& frameName, const mrpt::Clock::time_point& stamp,
    const mrpt::poses::CPose3D& poseInOdom)
{
    std::lock_guard<std::mutex> lck(mtx_);
    // Another source, or one that restarted: its poses do not continue the history.
    if (frameName != rawOdometryFrame_ ||
        (!rawOdometry_.empty() &&
         mrpt::system::timeDifference(rawOdometry_.back().stamp, stamp) <= 0))
    {
        rawOdometry_.clear();
        rawOdometryFrame_ = frameName;
    }
    rawOdometry_.push_back({stamp, poseInOdom});
    while (mrpt::system::timeDifference(rawOdometry_.front().stamp, stamp) > RAW_ODOMETRY_HISTORY)
    {
        rawOdometry_.pop_front();
    }
}

std::optional<mrpt::poses::CPose3D> FastPredictor::pose_through_odometry(
    const Snapshot& snap, const Parameters& params, const mrpt::Clock::time_point& t_query,
    const mrpt::math::TTwist3D& twist) const
{
    std::lock_guard<std::mutex> lck(mtx_);
    if (rawOdometry_.empty())
    {
        return std::nullopt;
    }
    const auto& str2id = snap.frameNames.getDirectMap();
    const auto  itName = str2id.find(rawOdometryFrame_);
    if (itName == str2id.end())
    {
        return std::nullopt;
    }
    const auto itFrame = snap.frameTransforms.find(itName->second);
    if (itFrame == snap.frameTransforms.end())
    {
        return std::nullopt;
    }
    const auto& newest = rawOdometry_.back();
    if (mrpt::system::timeDifference(rawOdometry_.front().stamp, t_query) < 0)
    {
        return std::nullopt;
    }

    mrpt::poses::CPose3D inOdom;
    if (const double past = mrpt::system::timeDifference(newest.stamp, t_query); past >= 0)
    {
        if (past > params.max_time_to_use_velocity_model)
        {
            return std::nullopt;
        }
        inOdom = newest.pose + body_twist_delta(params, twist, past);
    }
    else
    {
        // The first reading after t_query, and the one before it.
        const auto after = std::lower_bound(
            rawOdometry_.begin(), rawOdometry_.end(), t_query,
            [](const RawOdometry& r, const mrpt::Clock::time_point& t) { return r.stamp < t; });
        if (after == rawOdometry_.begin())
        {
            inOdom = after->pose;
        }
        else
        {
            const auto&  before = *std::prev(after);
            const double span   = mrpt::system::timeDifference(before.stamp, after->stamp);
            const double alpha  = mrpt::system::timeDifference(before.stamp, t_query) / span;
            auto         step   = mrpt::poses::Lie::SE<3>::log(after->pose - before.pose);
            step *= alpha;
            inOdom = before.pose + mrpt::poses::Lie::SE<3>::exp(step);
        }
    }
    return itFrame->second.mean + inOdom;
}

void FastPredictor::clear()
{
    std::lock_guard<std::mutex> lck(mtx_);
    snapshot_.reset();
    lastObsStamp_.reset();
    rawOdometry_.clear();
}

std::optional<NavState> FastPredictor::predict(
    const mrpt::Clock::time_point& t_query, const std::string& frame_id, const Parameters& params,
    mrpt::Clock::time_point* anchorStampOut) const
{
    std::shared_ptr<const Snapshot> snap;
    {
        std::lock_guard<std::mutex> lck(mtx_);
        snap = snapshot_;
    }
    if (!snap || !snap->valid)
    {
        return std::nullopt;
    }
    if (anchorStampOut)
    {
        *anchorStampOut = snap->anchorStamp;
    }

    // dt from the anchor (newest solved keyframe) to the requested time.
    const double dt = mrpt::system::timeDifference(snap->anchorStamp, t_query);
    if (std::abs(dt) > params.max_time_to_use_velocity_model)
    {
        return std::nullopt;
    }

    // Start from the anchor state and grow the twist uncertainty by the
    // random-walk model over the extrapolation interval (mirrors the
    // synchronous estimated_navstate()).
    NavState ret = snap->anchor;

    // The covariance propagation below (matrix inversions, information-form
    // conversions and Gaussian pose composition) can throw on a non
    // positive-definite covariance; estimated_navstate() delegates here directly
    // in async mode, so absorb it and report "not ready yet" instead of letting
    // it terminate the caller's thread.
    try
    {
        // Anchor twist covariance (before random-walk growth), reused as the
        // current-velocity uncertainty of the pose increment below.
        const mrpt::math::CMatrixDouble66 anchorTwistCov = snap->anchor.twist_inv_cov.inverse_LLt();
        {
            auto twist_cov = anchorTwistCov;
            for (int i = 0; i < 3; i++)
            {
                twist_cov(0 + i, 0 + i) +=
                    mrpt::square(params.sigma_random_walk_acceleration_linear * dt);
                twist_cov(3 + i, 3 + i) +=
                    mrpt::square(params.sigma_random_walk_acceleration_angular * dt);
            }
            ret.twist_inv_cov = twist_cov.inverse_LLt();
        }

        // Reference ({map}) frame: extrapolate the anchor pose forward,
        // propagating covariance.
        if (frame_id == params.reference_frame_name)
        {
            mrpt::poses::CPose3DPDFGaussian anchorPose;
            anchorPose.copyFrom(ret.pose);
            auto mapPdf = extrapolate_pose_pdf(params, anchorPose, ret.twist, anchorTwistCov, dt);
            // The uncertainty stays the extrapolated anchor's; where wheel
            // odometry is fused, the pose itself follows it (see the class docs).
            if (const auto followed = pose_through_odometry(*snap, params, t_query, ret.twist);
                followed)
            {
                mapPdf.mean = *followed;
            }
            apply_pose_sigma_floor(params, mapPdf);
            ret.pose.copyFrom(mapPdf);
            return ret;
        }

        // Odometry frame {odom_i}: resolve its numeric id.
        const auto& str2id = snap->frameNames.getDirectMap();
        const auto  itName = str2id.find(frame_id);
        if (itName == str2id.end())
        {
            return std::nullopt;
        }
        const auto requestedFrameIdx = itName->second;

        // Frame-local prediction: anchor on the source's OWN last raw pose in
        // {odom_i} and extrapolate by the body-twist increment, so the prediction
        // stays immune to {map} corrections (geo-ref / loop closure / per-solve
        // jitter). Falls back to the global conversion before the first raw pose.
        const auto itRaw = snap->lastRawPoseBySource.find(requestedFrameIdx);
        if (itRaw == snap->lastRawPoseBySource.end())
        {
            const auto itFrame = snap->frameTransforms.find(requestedFrameIdx);
            if (itFrame == snap->frameTransforms.end())
            {
                return std::nullopt;
            }
            mrpt::poses::CPose3DPDFGaussian anchorPose;
            anchorPose.copyFrom(ret.pose);
            const auto mapPred =
                extrapolate_pose_pdf(params, anchorPose, ret.twist, anchorTwistCov, dt);
            // Transform the {map}-frame prediction into {odom_i}: pred (-) T_frame_wrt_map.
            // The floor goes on AFTER the conversion, for the same reason as in
            // the synchronous path: it is a statement about the frame the caller
            // asked for.
            auto framePdf = mapPred - itFrame->second;
            apply_pose_sigma_floor(params, framePdf);
            ret.pose.copyFrom(framePdf);
            return ret;
        }

        const auto&  rawAnchor = itRaw->second;
        const double dtPred    = mrpt::system::timeDifference(rawAnchor.stamp, t_query);
        if (std::abs(dtPred) > params.max_time_to_use_velocity_model)
        {
            return std::nullopt;
        }

        auto rawPdf =
            extrapolate_pose_pdf(params, rawAnchor.pose, ret.twist, anchorTwistCov, dtPred);
        apply_pose_sigma_floor(params, rawPdf);
        ret.pose.copyFrom(rawPdf);

        return ret;
    }
    catch (const std::exception&)
    {
        // Under-constrained covariance: report "not ready yet".
        return std::nullopt;
    }
}

}  // namespace mola::state_estimation_smoother
