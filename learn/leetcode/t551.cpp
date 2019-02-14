#include <string>
using namespace std;
class Solution {
public:
  bool checkRecord(string s) {
    int A_cnt = 0, L_cnt = 0;
    for (auto &ss : s) {
      if (ss == 'A') {
        A_cnt++;
        L_cnt = 0;
      } else if (ss == 'L')
        L_cnt++;
      else {
        L_cnt = 0;
      }
      if (A_cnt > 1 || L_cnt > 2)
        return false;
    }
    return true;
  }
};
int main() {
  Solution s = Solution();
  bool result = s.checkRecord("PPALLPA");
  return 0;
}
