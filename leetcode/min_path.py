#
# 
# @param matrix int整型二维数组 the matrix
# @return int整型
# [[1,3,5,9],
#  [8,1,3,4],
#  [5,0,6,1],
#  [8,8,4,0]]
# 12
import math
class Solution:
    def minPathSum(self , matrix ):

        n_rows = len(matrix)
        n_cols = len(matrix[0])
        for i in range(n_rows):
            for j in range(n_cols):
                if i==0 and j!=0:
                    matrix[i][j] += matrix[i][j-1]
                elif i!=0 and j==0:
                    matrix[i][j] += matrix[i-1][j]
                elif i>0 and j>0:
                    matrix[i][j] += min(matrix[i][j-1],matrix[i-1][j])
        
        return matrix[n_rows-1][n_cols-1]
        
        
        
        
        
        
        
        
        
        
        
        