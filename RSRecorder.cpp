#include <sstream>

#include <Windows.h>

#include "RSRecorder.h"
#include "pxcfaceconfiguration.h"
#include "pxcfacemodule.h"
#include "pxcsensemanager.h"

#include <opencv2\opencv.hpp>

// Define to write CSV File
#include <ctime>
#include <fstream>
#include <iostream>

RSRecorder::~RSRecorder() {
  if (_sense_manager != NULL) _sense_manager->Release();
}
RSRecorder::RSRecorder()
    : _sense_manager(NULL),
      _face_data(NULL),
      _show_pose(false),
      _show_eyest(false),
      _show_ldmk(false),
      _frame_timestamp(0),
      _frame_id(0) {
  _sense_manager = PXCSenseManager::CreateInstance();
  _expression_map_init();
}

void RSRecorder::fini() {
  if (_face_data != NULL) _face_data->Release();
  if (_sense_manager != NULL) _sense_manager->Close();
  if (_ofs_pose.is_open()) _ofs_pose.close();
  if (_ofs_landmark.is_open()) _ofs_landmark.close();
  if (_video_writer.isOpened()) _video_writer.release();
  if (_ofs_expression.is_open()) _ofs_expression.close();
}

void RSRecorder::init(bool show_pose, bool show_eyest, bool show_ldmk) {
  _show_pose = show_pose;
  _show_eyest = show_eyest;
  _show_ldmk = show_ldmk;
  if (_sense_manager == NULL) {
    throw std::runtime_error("_sense_manager create fail");
  }
  PXCSession *session = _sense_manager->QuerySession();
  PXCSession::CoordinateSystem cs = session->QueryCoordinateSystem();
  std::cout << "COORDINATE SYSTEM " << cs << std::endl;

  std::time_t now_t = time(NULL);
  struct tm localt;
  localtime_s(&localt, &now_t);

  std::string newtime =
      std::to_string(localt.tm_year + 1900) + "_" +
      std::to_string(localt.tm_mon + 1) + "_" + std::to_string(localt.tm_mday) +
      "_" + std::to_string(localt.tm_hour) + "_" +
      std::to_string(localt.tm_min) + "_" + std::to_string(localt.tm_sec);

#pragma region VideoFileInit
  std::string videoname = "../videofiles/" + newtime + "_" + ".avi";
  cvNamedWindow("Record", CV_WINDOW_NORMAL);
  _video_writer.open(videoname, CV_FOURCC('X', 'V', 'I', 'D'), COLOR_FPS,
                     cvSize(COLOR_WIDTH, COLOR_HEIGHT), 1);

#pragma endregion VideoFileInit

#pragma region LandmarkCSVFileInit
  // Landmark first column
  _ofs_landmark.open("../csvfiles/" + newtime + "_landmark.csv");
  if (!_ofs_landmark.is_open())
    throw std::runtime_error("ofs landmark open fail");

  _ofs_landmark << "timestamp,frame";

  for (int s = 0; s < 78; s++) {
    _ofs_landmark << "," << s << "._image_x," << s << "._image_y";
  }

  _ofs_landmark << std::endl;

#pragma endregion LandmarkCSVFileInit

#pragma region ExpressionCSVFileInit

  // Expression
  _ofs_expression.open("../csvfiles/" + newtime + "_expression.csv");
  if (!_ofs_expression.is_open())
    throw std::runtime_error("ofs expression open fail");

  _ofs_expression << "timestamp,frame,"
                  << "Rect.x,Rect.y,Rect.w,Rect.h";
  for (auto expression_iter = _expression_map.begin();
       expression_iter != _expression_map.end(); expression_iter++) {
    _ofs_expression << "," << expression_iter->second.first;
  }

  _ofs_expression << std::endl;

#pragma endregion ExpressionCSVFileInit

#pragma region PoseCSVFileInit
  // Expression
  _ofs_pose.open("../csvfiles/" + newtime + "_pose.csv");
  if (!_ofs_pose.is_open()) throw std::runtime_error("ofs pose open fail");

  _ofs_pose << "timestamp,frame,"
            << "Rect.x,Rect.y,Rect.w,Rect.h,"
            << "yaw,"
            << "pitch,"
            << "roll" << std::endl;
#pragma endregion PoseCSVFileInit

  pxcStatus sts =
      _sense_manager->EnableStream(PXCCapture::StreamType::STREAM_TYPE_COLOR,
                                   COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS);
  if (sts < PXC_STATUS_NO_ERROR) {
    throw std::runtime_error("_sense_manager EnableStream fail");
  }

  _face_track_init();
}

void RSRecorder::_face_track_init() {
  pxcStatus sts = _sense_manager->EnableFace();
  if (sts < PXC_STATUS_NO_ERROR) {
    throw std::runtime_error("can't EnableFace");
  }

  /*

  sts = _sense_manager->EnableEmotion();
  if (sts<PXC_STATUS_NO_ERROR) {
  throw std::runtime_error("can't EnableEmotion");
  }
  */

  PXCFaceModule *faceModule = _sense_manager->QueryFace();
  if (faceModule == NULL) {
    throw std::runtime_error("QueryFace fail");
  }
  PXCFaceConfiguration *config = faceModule->CreateActiveConfiguration();
  if (config == NULL) {
    throw std::runtime_error("config create fail");
  }

  config->SetTrackingMode(
      PXCFaceConfiguration::TrackingModeType::FACE_MODE_COLOR_PLUS_DEPTH);
  config->ApplyChanges();

  sts = _sense_manager->Init();
  if (sts < PXC_STATUS_NO_ERROR) {
    throw std::runtime_error("sense manager init fail");
  }

  PXCCapture::Device *device =
      _sense_manager->QueryCaptureManager()->QueryDevice();
  if (device == NULL) {
    throw std::runtime_error("query device fail");
  }

  // device->SetMirrorMode(PXCCapture::Device::MirrorMode::MIRROR_MODE_HORIZONTAL);

  PXCCapture::DeviceInfo deviceInfo;
  device->QueryDeviceInfo(&deviceInfo);
  if (deviceInfo.model == PXCCapture::DEVICE_MODEL_IVCAM) {
    std::cout << "IS IVCAM" << std::endl;
    device->SetDepthConfidenceThreshold(1);
    device->SetIVCAMFilterOption(6);
    device->SetIVCAMMotionRangeTradeOff(21);
  }

  config->detection.isEnabled = true;
  config->detection.maxTrackedFaces = _MAXFACES;
  config->pose.isEnabled = true;
  config->pose.maxTrackedFaces = _MAXFACES;

  config->landmarks.isEnabled = true;
  config->landmarks.maxTrackedFaces = _MAXFACES;

  config->QueryExpressions()->Enable();
  config->QueryExpressions()->EnableAllExpressions();
  config->QueryExpressions()->properties.maxTrackedFaces = 2;

  config->ApplyChanges();

  _face_data = faceModule->CreateOutput();
}

void RSRecorder::run() {
  while (1) {
    _update_frame();
    _frame_id++;
    bool ret = _show_record_image();
    if (!ret) {
      break;
    }
  }
}

void RSRecorder::_update_frame() {
  pxcStatus sts = _sense_manager->AcquireFrame(true);
  if (sts < PXC_STATUS_NO_ERROR) {
    return;
  }

  _update_face();

  _sense_manager->ReleaseFrame();

  // showFps();
}

void RSRecorder::_update_face() {
  const PXCCapture::Sample *sample = _sense_manager->QuerySample();
  if (sample) {
    _update_mat(sample->color);
    _frame_timestamp = sample->color->QueryTimeStamp();
  }

  _face_data->Update();

  int numFaces = _face_data->QueryNumberOfDetectedFaces();
  if (numFaces > _MAXFACES) {
    std::cout << "face num " << numFaces << std::endl;
    numFaces = _MAXFACES;
  }

  for (int i = 0; i < numFaces; ++i) {
    auto face = _face_data->QueryFaceByIndex(i);
    if (face == 0) {
      continue;
    }

    PXCRectI32 faceRect = {0};
    auto detection = face->QueryDetection();
    if (detection != 0) {
      detection->QueryBoundingRect(&faceRect);
    }

    cv::rectangle(_show_mat,
                  cv::Rect(faceRect.x, faceRect.y, faceRect.w, faceRect.h),
                  cv::Scalar(255, 0, 0));
#pragma region GetFacePose

    PXCFaceData::PoseEulerAngles poseAngle[_MAXFACES] = {0};
    PXCFaceData::PoseData *pose = face->QueryPose();
    if (pose != NULL) {
      pxcBool sts = pose->QueryPoseAngles(&poseAngle[i]);
      if (sts < PXC_STATUS_NO_ERROR) {
        throw std::runtime_error("QueryPoseAngles failed");
      }
      // Writing to CSV files about face pose info
      _ofs_pose << _frame_timestamp << ","  //timestamp
                << _frame_id << ","         //frame
                << faceRect.x << "," << faceRect.y << "," << faceRect.w << ","
                << faceRect.h << "," << poseAngle[i].yaw << ","  //Angle
                << poseAngle[i].pitch << "," << poseAngle[i].roll << std::endl;
      if (_show_pose) {
        {
          std::stringstream ss;
          ss << "Yaw:" << poseAngle[i].yaw;
          cv::putText(_show_mat, ss.str(),
                      cv::Point(faceRect.x, faceRect.y - 75),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                      1, CV_AA);
        }

        {
          std::stringstream ss;
          ss << "Pitch:" << poseAngle[i].pitch;
          cv::putText(_show_mat, ss.str(),
                      cv::Point(faceRect.x, faceRect.y - 50),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                      1, CV_AA);
        }

        {
          std::stringstream ss;
          ss << "Roll:" << poseAngle[i].roll;
          cv::putText(_show_mat, ss.str(),
                      cv::Point(faceRect.x, faceRect.y - 25),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                      1, CV_AA);
        }
      }
    }
#pragma endregion GetFacePose

#pragma region GetFaceLandmarks

    PXCFaceData::LandmarksData *landmarkData[_MAXFACES];
    PXCFaceData::LandmarkPoint *landmarkPoints = nullptr;
    pxcI32 numPoints;

    landmarkData[i] = face->QueryLandmarks();
    if (landmarkData[i] != NULL) {
      numPoints = landmarkData[i]->QueryNumPoints();
      landmarkPoints = new PXCFaceData::LandmarkPoint[numPoints];

      if (landmarkData[i]->QueryPoints(landmarkPoints)) {
        _ofs_landmark << _frame_timestamp << ","  //timestamp
                      << _frame_id;               // frame

        for (int j = 0; j < numPoints; j++) {
          {
            std::stringstream ss;
            ss << j;

            if (/*landmarkPoints[j].source.alias != 0 &&*/ _show_ldmk) {
              if (landmarkPoints[j].confidenceImage > 50) {
                cv::circle(_show_mat,
                           cv::Point(landmarkPoints[j].image.x,
                                     landmarkPoints[j].image.y),
                           1, cv::Scalar(0, 255, 0), 1);
                cv::putText(_show_mat, ss.str(),
                            cv::Point(landmarkPoints[j].image.x,
                                      landmarkPoints[j].image.y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.2,
                            cv::Scalar(255, 255, 255), 1, CV_AA);
              } else
                cv::putText(_show_mat, ss.str(),
                            cv::Point(landmarkPoints[j].image.x,
                                      landmarkPoints[j].image.y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.2,
                            cv::Scalar(0, 0, 255), 1, CV_AA);
            }
          }
          // Writing to CSV files about Landmark info
          _ofs_landmark << "," << landmarkPoints[j].image.x << ","
                        << landmarkPoints[j].image.y;
        }
        _ofs_landmark << std::endl;
      }
      if (landmarkPoints) delete[] landmarkPoints;
    }

#pragma endregion GetFaceLandmarks

#pragma region GetFaceExpression

    PXCFaceData::ExpressionsData *expressionData;
    PXCFaceData::ExpressionsData::FaceExpressionResult expressionResult;
    int eye_closed_left = -1;
    int eye_closed_right = -1;
    expressionData = face->QueryExpressions();
    if (expressionData != NULL) {
      // Writing to CSV files about Landmark info
      _ofs_expression << _frame_timestamp << ","  //timestamp
                      << _frame_id << ","         //frame
                      << faceRect.x << "," << faceRect.y << "," << faceRect.w
                      << "," << faceRect.h;

      for (auto expression_iter = _expression_map.begin();
           expression_iter != _expression_map.end(); expression_iter++) {
        if (expressionData->QueryExpression(expression_iter->first,
                                            &expressionResult)) {
          expression_iter->second.second = expressionResult.intensity;
        }
        _ofs_expression << "," << expression_iter->second.second;
      }

      _ofs_expression << std::endl;
      if (_show_eyest) {
        {
          std::stringstream ss;
          ss << _expression_map
                    [PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_LEFT]
                        .first
             << ": "
             << _expression_map
                    [PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_LEFT]
                        .second;
          cv::putText(_show_mat, ss.str(),
                      cv::Point(faceRect.x, faceRect.y + faceRect.h + 20),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                      1, CV_AA);
        }
        {
          std::stringstream ss;
          ss << _expression_map
                    [PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_RIGHT]
                        .first
             << ": "
             << _expression_map
                    [PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_RIGHT]
                        .second;
          cv::putText(_show_mat, ss.str(),
                      cv::Point(faceRect.x, faceRect.y + faceRect.h + 45),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                      1, CV_AA);
        }
      }
    }
#pragma endregion GetFaceExpression

    {
      std::stringstream ss;
      ss << "Frame:" << _frame_id;
      cv::putText(_show_mat, ss.str(), cv::Point(50, 100),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2,
                  CV_AA);
    }
  }
}

void RSRecorder::_update_mat(PXCImage *colorFrame) {
  if (colorFrame == NULL) {
    return;
  }

  PXCImage::ImageInfo info = colorFrame->QueryInfo();

  PXCImage::ImageData data;
  pxcStatus sts = colorFrame->AcquireAccess(
      PXCImage::Access::ACCESS_READ, PXCImage::PixelFormat::PIXEL_FORMAT_RGB24,
      &data);
  if (sts < PXC_STATUS_NO_ERROR) {
    throw std::runtime_error("color access fail");
  }

  cv::Mat color_mat(info.height, info.width, CV_8UC3, data.planes[0]);
  _origin_mat = color_mat.clone();
  _show_mat = _origin_mat.clone();
  colorFrame->ReleaseAccess(&data);
}

bool RSRecorder::_show_record_image() {
  cv::imshow("show", _show_mat);
  _video_writer.write(_origin_mat);

  int c = cv::waitKey(1);
  if ((c == 27) || (c == 'q') || (c == 'Q')) {
    // ESC|q|Q for Exit
    return false;
  }

  return true;
}

void RSRecorder::_expression_map_init() {
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_SMILE] =
      std::make_pair("Smile", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_MOUTH_OPEN] =
      std::make_pair("Mouth Open", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_KISS] =
      std::make_pair("Kiss", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_EYES_TURN_LEFT] =
      std::make_pair("Eyes Turn Left", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_EYES_TURN_RIGHT] =
      std::make_pair("Eyes Turn Right", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_EYES_UP] =
      std::make_pair("Eyes Up", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_EYES_DOWN] =
      std::make_pair("Eyes Down", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_BROW_RAISER_LEFT] =
      std::make_pair("Brow Raised Left", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_BROW_RAISER_RIGHT] =
      std::make_pair("Brow Raised Right", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_BROW_LOWERER_LEFT] =
      std::make_pair("Brow Lowered Left", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_BROW_LOWERER_RIGHT] =
      std::make_pair("Brow Lowered Right", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_LEFT] =
      std::make_pair("Closed Eye Left", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_RIGHT] =
      std::make_pair("Closed Eye Right", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_TONGUE_OUT] =
      std::make_pair("Tongue Out", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_PUFF_RIGHT] =
      std::make_pair("Puff Right Cheek", -1);
  _expression_map[PXCFaceData::ExpressionsData::EXPRESSION_PUFF_LEFT] =
      std::make_pair("Puff Left Cheek", -1);
}

void RSRecorder::showFps() {
  static DWORD oldTime = ::timeGetTime();
  static int fps = 0;
  static int count = 0;

  count++;

  auto _new = ::timeGetTime();
  if ((_new - oldTime) >= 1000) {
    fps = count;
    count = 0;

    oldTime = _new;
  }

  std::stringstream ss;
  ss << "fps:" << fps;
  cv::putText(_show_mat, ss.str(), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX,
              1.2, cv::Scalar(0, 0, 255), 2, CV_AA);
}
