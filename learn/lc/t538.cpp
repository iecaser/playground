#include "common.hpp"
class Solution {
public:
  int val = 0;
  TreeNode *convertBST(TreeNode *root) {
    if (!root)
      return root;
    convertBST(root->right);
    val += root->val;
    root->val = val;
    convertBST(root->left);
    return root;
  }
};

int main() {
  Solution s = Solution();
  auto *root = new TreeNode(5);
  root->left = new TreeNode(2);
  root->right = new TreeNode(13);
  s.convertBST(root);
  return 0;
}
