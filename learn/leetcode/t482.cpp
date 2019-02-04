#include "common.hpp"

class Solution {
public:
  string licenseKeyFormatting(string S, int K) {
    string s = "";
    for (auto &ss : S) {
      if (ss != '-') {
        if ('a' <= ss && ss <= 'z') {
          s += (ss - 'a' + 'A');
        } else {
          s += ss;
        }
      }
    }
    string rtn = "";
    if (s.length() == 0)
      return rtn;
    int head = s.length() % K;
    rtn += s.substr(0, head);
    for (int i = head; i < s.length(); i += K) {
      rtn += "-" + s.substr(i, K);
    }
    if (head == 0)
      return rtn.substr(1, rtn.length());
    return rtn;
  }
};

// int main() {
//   Solution s = Solution();
//   string res = s.licenseKeyFormatting("---", 3);
//   cout << res << endl;
// }
