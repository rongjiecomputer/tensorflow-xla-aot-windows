#include <cstdio>
#include <algorithm>

#include "mnist.h"

int main() {
  MnistGraph mnist;

  float args[784] = {0};
  for (int i = 5; i < 28-5; i++) {
    for (int j = 28/2-2; j < 28/2+2; j++) {
      args[i*28 + j] = 0.8;
    }
  }
  std::copy(args, args + 784, mnist.arg0_data());

  if (!mnist.Run()) {
    puts(mnist.error_msg().c_str());
    return 1;
  }

  for (int i = 0; i < 10; i++) {
    printf("%d: %.lf\n", i, mnist.result0(0, i));
  }

  // Self-check
  const float* result = mnist.result0_data();
  auto max_ptr = std::max_element(result, result+10);
  auto max_index = std::distance(result, max_ptr);
  if (max_index == 1) {
    printf("Success\n");
  } else {
    printf("Failed. Expect '1' to have highest prediction score,\n"
           "but the model predicted it is '%d'.\n", static_cast<int>(max_index));
  }
  return 0;
}