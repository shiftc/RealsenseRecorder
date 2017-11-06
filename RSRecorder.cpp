#include <sstream>

#include <Windows.h>

#include "RSRecorder.h"
#include "pxcfaceconfiguration.h"
#include "pxcfacemodule.h"
#include "pxcsensemanager.h"

#include <opencv2\opencv.hpp>

// Define to write CSV File
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>

RSRecorder::~RSRecorder() {
  if (_sense_manager != NULL) {
    _sense_manager->Release();
  }
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
}

void RSRecorder::fini() {
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

  for (int s = 0; s < 78; s++) {
    _ofs_landmark << "," << s << "._world_x," << s << "._world_y," << s
                  << "._world_z";
  }

  _ofs_landmark << std::endl;

#pragma endregion LandmarkCSVFileInit

#pragma region ExpressionCSVFileInit

  // Expression
  _ofs_expression.open("../csvfiles/" + newtime + "_expression.csv");
  if (!_ofs_expression.is_open())
    throw std::runtime_error("ofs expression open fail");

  _ofs_expression << "timestamp,frame,"

                  << "Rect.x,Rect.y,Rect.w,Rect.h,"
                  << "BROW_RAISER_LEFT, BROW_RAISER_RIGHT, BROW_LOWERER_LEFT, "
                     "BROW_LOWERER_RIGHT,"
                  << "SMILE, KISS, MOUTH_OPEN, TONGUE_OUT,"
                  << "HEAD_TURN_LEFT, HEAD_TURN_RIGHT, HEAD_UP, HEAD_DOWN, "
                     "HEAD_TILT_LEFT,"
                  << "EYES_CLOSED_LEFT, EYES_CLOSED_RIGHT, EYES_TURN_LEFT, "
                     "EYES_TURN_RIGHT, EYES_UP, EYES_DOWN,"
                  << "PUFF_LEFT,PUFF_RIGHT," << std::endl;

#pragma endregion ExpressionCSVFileInit

#pragma region PoseCSVFileInit
  // Expression
  _ofs_pose.open("../csvfiles/" + newtime + "_pose.csv");
  if (!_ofs_pose.is_open()) throw std::runtime_error("ofs pose open fail");

  _ofs_pose << "timestamp,frame,"
            << "yaw,"
            << "pitch,"
            << "roll,"
            << "Rect.x,Rect.y,Rect.w,Rect.h," << std::endl;
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
    update_origin_mat(sample->color);
    _frame_timestamp = sample->color->QueryTimeStamp();
  }

  _face_data->Update();

  int numFaces = _face_data->QueryNumberOfDetectedFaces();
  if (numFaces > 1) {
    std::cout << "face num " << numFaces << std::endl;
    numFaces = 1;
  }

  for (int i = 0; i < numFaces; ++i) {
    auto face = _face_data->QueryFaceByIndex(i);
    if (face == 0) {
      continue;
    }

#pragma region GetFacePose

    PXCRectI32 faceRect = {0};
    PXCFaceData::PoseEulerAngles poseAngle[_MAXFACES] = {0};

    auto detection = face->QueryDetection();
    if (detection != 0) {
      detection->QueryBoundingRect(&faceRect);
    }

    cv::rectangle(_show_mat,
                  cv::Rect(faceRect.x, faceRect.y, faceRect.w, faceRect.h),
                  cv::Scalar(255, 0, 0));

    PXCFaceData::PoseData *pose = face->QueryPose();
    if (pose != NULL) {
      pxcBool sts = pose->QueryPoseAngles(&poseAngle[i]);
      if (sts < PXC_STATUS_NO_ERROR) {
        throw std::runtime_error("QueryPoseAngles failed");
      }
      _ofs_pose << _frame_timestamp << ","  //timestamp
                << _frame_id << ","         // frame
                << poseAngle[i].yaw << ","  //Angle
                << poseAngle[i].pitch << "," << poseAngle[i].roll << faceRect.x
                << "," << faceRect.y << "," << faceRect.w << "," << faceRect.h
                << std::endl;
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
              if (landmarkPoints[j].confidenceImage > 50)
                cv::putText(_show_mat, ss.str(),
                            cv::Point(landmarkPoints[j].image.x,
                                      landmarkPoints[j].image.y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.2,
                            cv::Scalar(255, 255, 255), 1, CV_AA);
              else
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

#if 0
          for(int s = 0; s < numPoints; s++) {
            pxcF32 land_x, land_y, land_z;

            if(fabsf(landmarkPoints[s].world.x) > 1)
              land_x = FloatRound(landmarkPoints[s].world.x, 4) * (-1) / 1000;
            else
              land_x = FloatRound(landmarkPoints[s].world.x, 4);

            if(fabsf(landmarkPoints[s].world.y) > 1)
              land_y = FloatRound(landmarkPoints[s].world.y, 4) / 1000;
            else
              land_y = FloatRound(landmarkPoints[s].world.y, 4);

            if(fabsf(landmarkPoints[s].world.z) > 2)
              land_z = FloatRound(landmarkPoints[s].world.z, 4) / 1000;
            else
              land_z = FloatRound(landmarkPoints[s].world.z, 4);

            _ofs_landmark << "," << FloatRound(land_x, 4) << ","
              << FloatRound(land_y, 4) << "," << FloatRound(land_z, 4);
          }
#endif
        }
        _ofs_landmark << std::endl;
        delete[] landmarkPoints;
      }
    }

#pragma endregion GetFaceLandmarks

#pragma region GetFaceExpression

    PXCFaceData::ExpressionsData *expressionData;
    PXCFaceData::ExpressionsData::FaceExpressionResult expressionResult;
    int expressionResult2[21];

    PXCFaceData::ExpressionsData::FaceExpression expressionLabel[21] = {
        PXCFaceData::ExpressionsData::EXPRESSION_BROW_RAISER_LEFT,
        PXCFaceData::ExpressionsData::EXPRESSION_BROW_RAISER_RIGHT,
        PXCFaceData::ExpressionsData::EXPRESSION_BROW_LOWERER_LEFT,
        PXCFaceData::ExpressionsData::EXPRESSION_BROW_LOWERER_RIGHT,
        PXCFaceData::ExpressionsData::EXPRESSION_SMILE,
        PXCFaceData::ExpressionsData::EXPRESSION_KISS,
        PXCFaceData::ExpressionsData::EXPRESSION_MOUTH_OPEN,
        PXCFaceData::ExpressionsData::EXPRESSION_TONGUE_OUT,
        PXCFaceData::ExpressionsData::EXPRESSION_HEAD_TURN_LEFT,
        PXCFaceData::ExpressionsData::EXPRESSION_HEAD_TURN_RIGHT,
        PXCFaceData::ExpressionsData::EXPRESSION_HEAD_UP,
        PXCFaceData::ExpressionsData::EXPRESSION_HEAD_DOWN,
        PXCFaceData::ExpressionsData::EXPRESSION_HEAD_TILT_LEFT,
        PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_LEFT,
        PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_RIGHT,
        PXCFaceData::ExpressionsData::EXPRESSION_EYES_TURN_LEFT,
        PXCFaceData::ExpressionsData::EXPRESSION_EYES_TURN_RIGHT,
        PXCFaceData::ExpressionsData::EXPRESSION_EYES_UP,
        PXCFaceData::ExpressionsData::EXPRESSION_EYES_DOWN,
        PXCFaceData::ExpressionsData::EXPRESSION_PUFF_LEFT,
        PXCFaceData::ExpressionsData::EXPRESSION_PUFF_RIGHT};

    expressionData = face->QueryExpressions();
    if (expressionData != NULL) {
      for (int jj = 0; jj < 21; jj++) {
        expressionResult2[jj] = 0;
        if (expressionData->QueryExpression(expressionLabel[jj],
                                            &expressionResult)) {
          {
            expressionResult2[jj] = expressionResult.intensity;
          }
        }
      }

      if (_show_eyest) {
        int eye_turn_left = -1;
        int eye_turn_right = -1;
        int eye_up = -1;
        int eye_down = -1;
        int eye_closed_left = -1;
        int eye_closed_right = -1;

        if (expressionData->QueryExpression(
                PXCFaceData::ExpressionsData::EXPRESSION_EYES_TURN_LEFT,
                &expressionResult)) {
          eye_turn_left = expressionResult.intensity;
        }
        if (expressionData->QueryExpression(
                PXCFaceData::ExpressionsData::EXPRESSION_EYES_TURN_RIGHT,
                &expressionResult)) {
          eye_turn_right = expressionResult.intensity;
        }
        if (expressionData->QueryExpression(
                PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_LEFT,
                &expressionResult)) {
          eye_closed_left = expressionResult.intensity;
        }
        if (expressionData->QueryExpression(
                PXCFaceData::ExpressionsData::EXPRESSION_EYES_CLOSED_RIGHT,
                &expressionResult)) {
          eye_closed_right = expressionResult.intensity;
        }
        if (expressionData->QueryExpression(
                PXCFaceData::ExpressionsData::EXPRESSION_EYES_UP,
                &expressionResult)) {
          eye_up = expressionResult.intensity;
        }
        if (expressionData->QueryExpression(
                PXCFaceData::ExpressionsData::EXPRESSION_EYES_DOWN,
                &expressionResult)) {
          eye_down = expressionResult.intensity;
        }

        int eyeX;
        int eyeY;

        if (eye_turn_left == eye_turn_right == -1) {
          eyeX = -9999;
        } else if (eye_turn_left == eye_turn_right) {
          eyeX = eye_turn_left;
        } else if (eye_turn_left < eye_turn_right) {
          eyeX = eye_turn_right;
        } else {
          eyeX = -eye_turn_left;
        }

        if (eye_up == eye_down == -1) {
          eyeY = -9999;
        } else if (eye_up == eye_down) {
          eyeY = eye_up;
        } else if (eye_up > eye_down) {
          eyeY = eye_up;
        } else {
          eyeY = -eye_down;
        }

        std::stringstream ss1;
        ss1 << "EYE_LOCATION: " << eyeX << ", " << eyeY;
        cv::putText(_show_mat, ss1.str(),
                    cv::Point(faceRect.x, faceRect.y + faceRect.h + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2,
                    CV_AA);

        std::stringstream ss2;
        ss2 << "EYE_CLOSED: " << eye_closed_left << ", " << eye_closed_right;
        cv::putText(_show_mat, ss2.str(),
                    cv::Point(faceRect.x, faceRect.y + faceRect.h + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2,
                    CV_AA);
      }

      // Writing to CSV files about Landmark info
      _ofs_expression << _frame_timestamp << ","  //timestamp
                      << _frame_id << ","         //frame
                      << faceRect.x << "," << faceRect.y << "," << faceRect.w
                      << "," << faceRect.h << ",";

      for (int jj = 0; jj < 21; jj++) {
        _ofs_expression << expressionResult2[jj];
        if (jj < 20) {
          _ofs_expression << ",";
        }
      }

      _ofs_expression << std::endl;
    }
#pragma endregion GetFaceExpression

#pragma region WriteCSVFiles

    {
      std::stringstream ss;
      ss << "Frame:" << _frame_id;
      cv::putText(_show_mat, ss.str(), cv::Point(50, 100),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2,
                  CV_AA);
    }

#pragma endregion WriteCSVFiles
  }
}

pxcF32 RSRecorder::FloatRound(pxcF32 f, int roundNum) {
  pxcF32 ret = 0;
  int _r;
  if (f == 0) {
    return 0;
  } else {
    _r = (int)(f * (pow(10, roundNum)));

    ret = _r / pow(10, roundNum);

    return ret;
  }
}

void RSRecorder::update_origin_mat(PXCImage *colorFrame) {
  if (colorFrame == 0) {
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
  //_video_writer.write(_origin_mat);

  int c = cv::waitKey(1);
  if ((c == 27) || (c == 'q') || (c == 'Q')) {
    // ESC|q|Q for Exit
    return false;
  }

  return true;
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
