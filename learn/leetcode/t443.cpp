#include "common.hpp"

class Solution {
private:
  string numToString(int num) {
    string res = "";
    if (num == 1)
      return res;
    return to_string(num);
  }

public:
  int compress(vector<char> &chars) {
    int res = 0;
    char cPrev = chars[0];
    int charCount = 1;
    string str;
    for (int i = 1; i < chars.size(); ++i) {
      if (cPrev == chars[i]) {
        charCount++;
      } else {
        str = numToString(charCount);
        str = cPrev + str;
        for (auto &s : str)
          chars[res++] = s;
        cPrev = chars[i];
        charCount = 1;
      }
    }
    str = numToString(charCount);
    str = cPrev + str;
    for (auto &s : str)
      chars[res++] = s;
    return res;
  }
};

int main() {
  vector<char> chars = {'a', 'a', 'b', 'b', 'b'};
  Solution s = Solution();
  int res = s.compress(chars);
  cout << res << endl;
  return 0;
}
