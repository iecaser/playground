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
class Solution {
public:
  Node *construct(vector<vector<int> > &grid) {}
  Node *build(vector<vector<int> > &grid, int x, int y, int size) {
    if (size == 1)
      return new Node(grid[x][y] == 1, true, nullptr, nullptr, nullptr,
                      nullptr);
  }
};
