# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

#
# 
# @param root TreeNode类 the root of binary tree
# @return int整型二维数组
#
class Solution:
    def rec(self,root,flag):
        if root!=None:
            if flag==1:
                tmp = [root.val]+self.rec(root.left,flag)+self.rec(root.right,flag)
                return tmp
            if flag==2:
                tmp = self.rec(root.left,flag)+[root.val]+self.rec(root.right,flag)
                return tmp
            if flag==3:
                tmp = self.rec(root.left,flag)+self.rec(root.right,flag)+[root.val]
                return tmp
        return []
    
    def threeOrders(self , root ):
        # write code here
        # 前序 root-left-right
        # 中序 left-root-right
        # 后序 left-right-root
        
        return [self.rec(root,1),
                self.rec(root,2),
                self.rec(root,3)]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        