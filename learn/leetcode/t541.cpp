#include <string>
using namespace std;
class Solution {
public:
  string reverseStr(string s, int k) {
    for (int start = 0; start < s.length(); start += 2 * k) {
      int i = start, j = start + k - 1;
      if (j > s.length() - 1)
        j = s.length() - 1;
      while (i < j) {
        char tmp = s[i];
        s[i++] = s[j];
        s[j--] = tmp;
      }
    }
    return s;
  }
};

int main() {
  string str = "a";
  int k = 2;
  Solution s = Solution();
  string result = s.reverseStr(str, k);
}
