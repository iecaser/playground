#include "common.hpp"

class Solution1 {
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
class Solution {
public:
  int compress(vector<char> &chars) {
    int anchor = 0, write = 0;
    for (int i = 0; i < chars.size(); ++i) {
      if (chars[i] != chars[i + 1] || i == chars.size() - 1) {
        chars[write++] = chars[i];
        if (i > anchor)
          for (auto &c : to_string(i - anchor + 1))
            chars[write++] = c;
        anchor = i + 1;
      }
    }
    return write;
  }
};

// int main() {
//   vector<char> chars = {'a', 'a', 'b', 'b', 'b'};
//   Solution s = Solution();
//   int res = s.compress(chars);
//   cout << res << endl;
//   return 0;
// }
