#include <cassert.h>
#include <iostream>
using namespace std;

int main() {

#ifndef NDEBUG
  assert(true);
#endif
  cout << "hello world" << endl;
  return 0;
}
