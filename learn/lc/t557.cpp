# include<string>
# include<iostream>

using namespace std;

class Solution {
public:
    string reverseWords(string s) {
        int head = 0;
        char tmp;
        for (int i = 0; i <= s.length(); ++i) {
            if (i == s.length() || s[i] == ' ') {
                for (int j = 0; j < (i - head + 1) / 2; ++j) {
                    tmp = s[j + head];
                    s[j + head] = s[i - 1 - j];
                    s[i - 1 - j] = tmp;
                }
                head = i + 1;
            }
        }
        return s;
    }
};

int main() {
    Solution sln = Solution();
    string s = "hello world";
    string ss = sln.reverseWords(s);
    cout << s << endl;
    cout << ss << endl;
    return 0;
}
