#include "common.hpp"

class Solution {
public:
  int poorPigs(int buckets, int minutesToDie, int minutesToTest) {
    int round = minutesToTest / minutesToDie;
    int i = 1, count = 0;
    while (i < buckets) {
      i *= (round + 1);
      count++;
    }
    return count;
  }
};

// int main() {
//   Solution s = Solution();
//   int res = s.poorPigs(1000, 12, 60);
//   cout << res << endl;
//   return 0;
// }
