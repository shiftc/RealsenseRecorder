
#include "RSRecorder.h"
#include <iostream>

void main() {
  try {
    RSRecorder rs_recorder;
    rs_recorder.init(true, true, true);
    rs_recorder.run();
    rs_recorder.fini();
  } catch (std::exception &ex) {
    std::cout << ex.what() << std::endl;
  }
}
