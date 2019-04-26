#include "common.hpp"
class Solution {
public:
  int maxlen = 0;
  int max(int a, int b) { return a > b ? a : b; }
  int helper(TreeNode *root) {
    if (!root)
      return 0;
    int left = 0, right = 0;
    if (root->left)
      left = helper(root->left) + 1;
    if (root->right)
      right = helper(root->right) + 1;
    maxlen = max(maxlen, left + right);
    return max(left, right);
  }
  int diameterOfBinaryTree(TreeNode *root) {
    helper(root);
    return maxlen;
  }
};
int main() {
  Solution s = Solution();
  TreeNode *root = new TreeNode(1);
  root->left = new TreeNode(2);
  root->right = new TreeNode(3);
  root->left->left = new TreeNode(4);
  root->left->right = new TreeNode(5);
  int result = s.diameterOfBinaryTree(root);
  return 0;
}
