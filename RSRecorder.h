#pragma once
#include "pxcfaceconfiguration.h"
#include "pxcfacemodule.h"
#include "pxcsensemanager.h"

#include <opencv2\opencv.hpp>
#include <string>
class RSRecorder {
public:
  ~RSRecorder();
  RSRecorder();

  void fini();
  void init(bool show_pose = true, bool show_eyest = false, bool show_ldmk = false);
  void run();

private:
  void _face_track_init();
  void _update_frame();

  void _update_face();

  pxcF32 FloatRound(pxcF32 f, int roundNum);

  void update_origin_mat(PXCImage *colorFrame);

  bool _show_record_image();

  void showFps();

private:
  // cv::Mat _origin_mat;
  cv::Mat _origin_mat;
  cv::Mat _show_mat;
  PXCSenseManager *_sense_manager;
  PXCFaceData *_face_data;
  cv::VideoWriter _video_writer;
  std::ofstream _ofs_landmark;
  std::ofstream _ofs_expression;
  std::ofstream _ofs_pose;


  bool _show_pose;
  bool _show_eyest;
  bool _show_ldmk;

  static const int COLOR_WIDTH = 1280;
  static const int COLOR_HEIGHT = 720;
  static const int COLOR_FPS = 30;
  static const int _MAXFACES = 1;

  pxcI64 _frame_timestamp;
  int64 _frame_id;


  errno_t err;


};