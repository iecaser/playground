#include <vector>
using namespace std;

// Definition for a QuadTree node.
class Node {
public:
  bool val;
  bool isLeaf;
  Node *topLeft;
  Node *topRight;
  Node *bottomLeft;
  Node *bottomRight;

  Node() {}

  Node(bool _val, bool _isLeaf, Node *_topLeft, Node *_topRight,
       Node *_bottomLeft, Node *_bottomRight) {
    val = _val;
    isLeaf = _isLeaf;
    topLeft = _topLeft;
    topRight = _topRight;
    bottomLeft = _bottomLeft;
    bottomRight = _bottomRight;
  }
};
class Solution1 {
public:
  Node *build(vector<vector<int>> &grid, int x, int y, int size) {
    if (size == 1)
      return new Node(grid[x][y] == 1, true, nullptr, nullptr, nullptr,
                      nullptr);
    int newSize = size / 2;
    Node *topLeft = build(grid, x, y, newSize);
    Node *topRight = build(grid, x, y + newSize, newSize);
    Node *bottomLeft = build(grid, x + newSize, y, newSize);
    Node *bottomRight = build(grid, x + newSize, y + newSize, newSize);
    bool isAllOne =
        topLeft->val && topRight->val && bottomLeft->val && bottomRight->val;
    bool isAllZero = !topLeft->val && !topRight->val && !bottomLeft->val &&
                     !bottomRight->val;
    bool isAllLeaf = topLeft->isLeaf && topRight->isLeaf &&
                     bottomLeft->isLeaf && bottomRight->isLeaf;
    int val = topLeft->val;
    if (isAllLeaf && (isAllOne || isAllZero)) {
      delete topLeft;
      topLeft = nullptr;
      delete topRight;
      topRight = nullptr;
      delete bottomLeft;
      bottomLeft = nullptr;
      delete bottomRight;
      bottomRight = nullptr;
      return new Node(val, true, nullptr, nullptr, nullptr, nullptr);
    } else {
      return new Node(val, false, topLeft, topRight, bottomLeft, bottomRight);
    }
  }
  Node *construct(vector<vector<int>> &grid) {
    int N = grid.size();
    if (N == 0)
      return nullptr;
    return build(grid, 0, 0, N);
  }
};

class Solution {
  Node *build(vector<vector<int>> &grid, int x, int y, int size) {
    if (size == 1)
      return new Node(grid[x][y], true, nullptr, nullptr, nullptr, nullptr);
    int val = grid[x][y];
    Node *root = new Node(val, true, nullptr, nullptr, nullptr, nullptr);
    bool isLeaf = true;
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j)
        if (grid[x + i][y + j] != val) {
          isLeaf = false;
          break;
        }
    if (!isLeaf) {
      size /= 2;
      Node *topLeft = build(grid, x, y, size);
      Node *topRight = build(grid, x, y + size, size);
      Node *bottomLeft = build(grid, x + size, y, size);
      Node *bottomRight = build(grid, x + size, y + size, size);
      return new Node(false, false, topLeft, topRight, bottomLeft, bottomRight);
    }
    return root;
  }

public:
  Node *construct(vector<vector<int>> &grid) {
    int N = grid.size();
    if (N == 0)
      return nullptr;
    return build(grid, 0, 0, N);
  }
};

// test
// int main() {
//   vector<vector<int>> grid = {{1, 0}, {1, 1}};
//   Solution *s = new Solution();
//   Node *root = s->construct(grid);
//   return 0;
// }
