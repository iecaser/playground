#include "common.hpp"

class Node {
public:
  int val;
  vector<Node *> children;

  Node() {}

  Node(int _val, vector<Node *> _children) {
    val = _val;
    children = _children;
  }
};

class Solution {
public:
  vector<vector<int>> levelOrder(Node *root) {
    vector<vector<int>> res;
    if (!root)
      return res;
    queue<Node *> q;
    q.push(root);
    while (!q.empty()) {
      int size = q.size();
      vector<int> levelRes;
      while (size--) {
        root = q.front();
        q.pop();
        levelRes.push_back(root->val);
        for (auto child : root->children)
          q.push(child);
      }
      res.push_back(levelRes);
    }
    return res;
  }
};
