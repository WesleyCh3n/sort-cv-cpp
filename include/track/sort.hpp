#pragma once

#include <opencv2/core/types.hpp>
#include <set>
#include <vector>

#include "hungarian.hpp"
#include "kalman.hpp"

namespace track {
template <class T> struct TrackingBox {
  int id;
  cv::Rect_<T> box;
};

template <class T>
inline double GetIOU(cv::Rect_<T> bb_test, cv::Rect_<T> bb_gt) {
  float in = (bb_test & bb_gt).area();
  float un = bb_test.area() + bb_gt.area() - in;

  if (un < DBL_EPSILON)
    return 0;

  return (double)(in / un);
}

template <class T> class SORT {
  // Maximum number of frames to keep alive a track without associated
  // detections.
  int max_age_;
  // Minimum number of associated detections before track is initialised.
  int min_hits_;
  // Minimum IOU for match.
  double iou_threshold_;

  std::vector<KalmanTracker<T>> kfs_;
  std::vector<cv::Rect_<T>> kf_predicts_;
  std::vector<std::vector<double>> iou_matrix_;
  std::vector<int> assignment_;
  std::set<int> unmatch_detection_;
  std::set<int> unmatched_trajectories_;
  std::set<int> all_items_;
  std::set<int> matched_items_;
  std::vector<cv::Point> matched_pairs_;
  std::vector<TrackingBox<T>> frameTrackingResult;

  // this is not real count, just for evalurate frame_count_ <= min_hits_
  int frame_count_ = 0;
  uint64_t track_id = 0;

  std::vector<uint64_t> removed_ids_;

public:
  SORT(int max_age, int min_hits, double iou_threshold)
      : max_age_(max_age), min_hits_(min_hits), iou_threshold_(iou_threshold) {}

  std::vector<TrackingBox<T>>
  predict_with_removed(const std::vector<cv::Rect_<T>> &detections,
                       std::vector<uint64_t> &removed_ids) {
    auto results = predict(detections);
    removed_ids = std::move(removed_ids_); // move out the removed ids
    return results;
  }

  std::vector<TrackingBox<T>>
  predict(const std::vector<cv::Rect_<T>> &detections) {
    // prevent track id overflow
    if (track_id == UINT64_MAX - 1) {
      track_id = 0;
    }
    // NOTE: you can comment if statement if you want real frame count
    if (frame_count_ <= min_hits_)
      frame_count_++;
    // empty removed id
    removed_ids_.clear();

    // if no tracking object is alive, or first frame met,
    if (kfs_.size() == 0) {
      // reset track_id
      track_id = 0;
      // initialize kalman trackers using first detections.
      for (unsigned int i = 0; i < detections.size(); i++) {
        KalmanTracker trk = KalmanTracker(detections[i], track_id++);
        kfs_.emplace_back(trk);
      }
      // output the first frame detections
      frameTrackingResult.clear();
      for (unsigned int id = 0; id < detections.size(); id++) {
        TrackingBox<T> res;
        res.box = detections[id];
        res.id = id;
        frameTrackingResult.push_back(res);
      }
      return frameTrackingResult;
    }

    ///////////////////////////////////////
    // 3.1. get predicted locations from existing trackers.
    kf_predicts_.clear();

    for (auto it = kfs_.begin(); it != kfs_.end();) {
      cv::Rect_<T> pBox = (*it).predict();
      if (pBox.x >= 0 && pBox.y >= 0) {
        kf_predicts_.push_back(pBox);
        it++;
      } else {
        removed_ids_.emplace_back((*it).m_id);
        it = kfs_.erase(it);
      }
    }

    ///////////////////////////////////////
    // 3.2. associate detections to tracked object (both represented as bounding
    // boxes) dets : detFrameData[fi]
    unsigned int track_num = kf_predicts_.size();
    unsigned int detection_num = detections.size();

    iou_matrix_.clear();
    iou_matrix_.resize(track_num, vector<double>(detection_num, 0));

    for (unsigned int i = 0; i < track_num;
         i++) // compute iou matrix as a distance matrix
    {
      for (unsigned int j = 0; j < detection_num; j++) {
        // use 1-iou because the hungarian algorithm computes a minimum-cost
        // assignment.
        iou_matrix_[i][j] = 1 - GetIOU(kf_predicts_[i], detections[j]);
      }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with
    // len=preNum
    HungarianAlgorithm HungAlgo;
    assignment_.clear();
    HungAlgo.Solve(iou_matrix_, assignment_);

    // find matches, unmatched_detections and unmatched_predictions
    unmatched_trajectories_.clear();
    unmatch_detection_.clear();
    all_items_.clear();
    matched_items_.clear();

    if (detection_num > track_num) //	there are unmatched detections
    {
      for (unsigned int n = 0; n < detection_num; n++)
        all_items_.insert(n);

      for (unsigned int i = 0; i < track_num; ++i)
        matched_items_.insert(assignment_[i]);

      set_difference(all_items_.begin(), all_items_.end(),
                     matched_items_.begin(), matched_items_.end(),
                     insert_iterator<set<int>>(unmatch_detection_,
                                               unmatch_detection_.begin()));
    } else if (detection_num <
               track_num) // there are unmatched trajectory/predictions
    {
      for (unsigned int i = 0; i < track_num; ++i)
        if (assignment_[i] == -1) // unassigned label will be set as -1 in the
                                  // assignment algorithm
          unmatched_trajectories_.insert(i);
    } else {
    }

    // filter out matched with low IOU
    matched_pairs_.clear();
    for (unsigned int i = 0; i < track_num; ++i) {
      if (assignment_[i] == -1) // pass over invalid values
        continue;
      if (1 - iou_matrix_[i][assignment_[i]] < iou_threshold_) {
        unmatched_trajectories_.insert(i);
        unmatch_detection_.insert(assignment_[i]);
      } else
        matched_pairs_.push_back(cv::Point(i, assignment_[i]));
    }

    ///////////////////////////////////////
    // 3.3. updating trackers

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matched_pairs_.size(); i++) {
      trkIdx = matched_pairs_[i].x;
      detIdx = matched_pairs_[i].y;
      kfs_[trkIdx].update(detections[detIdx]);
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatch_detection_) {
      KalmanTracker tracker = KalmanTracker(detections[umd], track_id++);
      kfs_.emplace_back(tracker);
    }

    // get trackers' output
    frameTrackingResult.clear();
    for (auto it = kfs_.begin(); it != kfs_.end();) {
      if (((*it).m_time_since_update < 1) &&
          ((*it).m_hit_streak >= min_hits_ || frame_count_ <= min_hits_)) {
        TrackingBox<T> res;
        res.box = (*it).get_state();
        res.id = (*it).m_id;
        frameTrackingResult.push_back(res);
        it++;
      } else
        it++;

      // remove dead tracklet
      if (it != kfs_.end() && (*it).m_time_since_update > max_age_) {
        removed_ids_.emplace_back((*it).m_id);
        it = kfs_.erase(it);
      }
    }
    return frameTrackingResult;
  }
};
} // namespace track
