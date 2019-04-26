#include "common.hpp"

class Solution1 {
public:
  int fib(int N) {
    if (N <= 0)
      return 0;
    if (N <= 2)
      return 1;
    return fib(N - 1) + fib(N - 2);
  }
};
class Solution {
public:
  int fib(int N) {
    int f0 = 1, f1 = 0, f2 = 0;
    for (int i = 0; i < N; ++i) {
      f2 = f0 + f1;
      f0 = f1;
      f1 = f2;
    }
    return f2;
  }
};
// int main() {
//   Solution s = Solution();
//   cout << s.fib(4) << endl;
// }
