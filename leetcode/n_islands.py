#
# 判断岛屿数量
# @param grid char字符型二维数组 
# @return int整型
# [[1,1,0,0,0],
#  [0,1,0,1,1],
#  [0,0,0,1,1],
#  [0,0,0,0,0],
#  [0,0,1,1,1]]
# 3
class Solution:
    def rec(self, grid, grid_map, i, j, k):
        n_rows = len(grid)
        n_cols = len(grid[0])
        
        if i+1<n_rows:
            if grid[i+1][j]=='1' and grid_map[i+1][j]==0:
                grid_map[i+1][j]=k
                self.rec(grid,grid_map,i+1,j,k)
            
        if i-1>=0:
            if grid[i-1][j]=='1' and grid_map[i-1][j]==0:
                grid_map[i-1][j]=k
                self.rec(grid,grid_map,i-1,j,k)
            
        if j-1>=0:
            if grid[i][j-1]=='1' and grid_map[i][j-1]==0:
                grid_map[i][j-1]=k
                self.rec(grid,grid_map,i,j-1,k)
                
        if j+1<n_cols:
            if grid[i][j+1]=='1' and grid_map[i][j+1]==0:
                grid_map[i][j+1]=k
                self.rec(grid,grid_map,i,j+1,k)
        return
        
        
    def solve(self , grid ):
        # write code here
        n_islands = 0
        n_rows = len(grid)
        n_cols = len(grid[0])
        grid_map = [[0 for i in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                if grid[i][j]=='1' and grid_map[i][j]==0:
                    n_islands+=1
                    self.rec(grid,grid_map,i,j,n_islands)
                    
        return n_islands
                    
                    
                    
                    
                    
                    
                    