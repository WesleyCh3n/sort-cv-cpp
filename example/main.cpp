#include <assert.h>
#include <fstream>
#include <iostream>

#include "track/sort.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <unordered_map>

template <class T> struct DetectionRect {
  int frame;
  int id;
  cv::Rect_<T> box;
};

void TestSORT(string seqName, bool display);

int main() {
  vector<string> sequences = {
      "ADL-Rundle-6", //
                      // "ADL-Rundle-8", //
                      // "ETH-Bahnhof",    //
                      // "ETH-Pedcross2",  //
                      // "ETH-Sunnyday",   //
                      // "KITTI-13",       //
                      // "KITTI-17",       //
                      // "TUD-Campus",     //
                      // "TUD-Stadtmitte", //
                      // "Venice-2",       //
                      // "PETS09-S2L1",    //
  };
  for (auto seq : sequences)
    TestSORT(seq, false);

  return 0;
}

void TestSORT(string seqName, bool display) {
  // 1. read detection file
  ifstream det_file;
  string det_file_path = "data/" + seqName + "/det.txt";
  det_file.open(det_file_path);

  if (!det_file.is_open()) {
    cerr << "Error: can not find file " << det_file_path << endl;
    return;
  }

  string line;
  istringstream ss;
  vector<DetectionRect<float>> data;
  char ch;
  float tpx, tpy, tpw, tph;
  int max_frame = 0;
  while (getline(det_file, line)) {
    DetectionRect<float> tb;
    ss.str(line);
    ss >> tb.frame >> ch >> tb.id >> ch;
    ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
    ss.str("");
    max_frame = max(max_frame, tb.frame);

    tb.box = cv::Rect_<float>(cv::Point_<float>(tpx, tpy),
                              cv::Point_<float>(tpx + tpw, tpy + tph));
    data.push_back(tb);
  }
  det_file.close();

  // 2. group detData by frame
  std::vector<std::vector<cv::Rect_<float>>> det_frames;
  for (int i = 0; i < max_frame; i++) {
    std::vector<cv::Rect_<float>> tmp;
    for (auto f : data)
      if (f.frame == i + 1) { // frame num starts from 1
        tmp.emplace_back(f.box);
      }
    det_frames.emplace_back(tmp);
  }

  // 3. run SORT tracking on each frame
  track::SORT<float> sort_algo(1, 1, 0.3);
  std::vector<uint64_t> removed_ids;
  std::unordered_map<uint64_t, int> trk_ids;
  for (int fi = 0; fi < det_frames.size(); fi++) {
    std::cout << "=========\n";
    auto res = sort_algo.predict_with_removed(det_frames[fi], removed_ids);

    fmt::print("track: ");
    for (auto tb : res)
      fmt::print("{} ", tb.id);
    fmt::println("");
    fmt::println("Removed ids: {}", removed_ids);

    for (const auto &tb : res) {
      if (trk_ids.count(tb.id) == 0) {
        trk_ids[tb.id] = fi;
      }
    }
    for (const auto &id : removed_ids) {
      if (auto it = trk_ids.find(id); it != trk_ids.end()) {
        fmt::println("Removed {} with {}", id, it->second);
        trk_ids.erase(it);
      }
    }
    fmt::println("trk_ids size: {}, res size: {}", trk_ids.size(), res.size());
    fmt::println("trk_ids: {}", trk_ids);
    // for (auto tb : res)
    //   std::cout << "x"
    //             << "," << tb.id << "," << tb.box.x << "," << tb.box.y << ","
    //             << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1"
    //             << std::endl;
  }
}
