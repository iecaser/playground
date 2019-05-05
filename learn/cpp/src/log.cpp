#include <cassert>
#include <iostream>
using namespace std;
// #define NDEBUG

int main() {

#ifndef NDEBUG
  cerr << "file: " << __FILE__ << endl;
  cerr << "func: " << __func__ << endl;
  cerr << "line: " << __LINE__ << endl;
  cerr << "time: " << __TIME__ << endl;
  cerr << "data: " << __DATE__ << endl;
  assert(false);
#endif
  cout << "hello world" << endl;
  return 0;
}
