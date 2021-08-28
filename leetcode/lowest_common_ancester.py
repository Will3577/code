class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

#
# 
# @param root TreeNode类 
# @param o1 int整型 
# @param o2 int整型 
# @return int整型
# [3,5,1,6,2,0,8,#,#,7,4],5,1
# 3
class Solution:
    def rec(self, root, o1, o2):
        if root==None or root.val==o1 or root.val==o2:
            return root
        
        left = self.rec(root.left,o1,o2)
        right = self.rec(root.right,o1,o2)
        
        if left==None:
            return right
        if right==None:
            return left
        return root
    def lowestCommonAncestor(self , root , o1 , o2 ):
        # write code here
        return self.rec(root,o1,o2).val
        
        
        
        
        
        
        
        
        
        
        