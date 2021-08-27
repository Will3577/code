#
# retrun the longest increasing subsequence
# @param arr int整型一维数组 the array
# @return int整型一维数组
#

# Time Complexity: O(nlogn)
import math
class Solution:
    def binary_S(self,x, dp):
        left = 0
        right = len(dp)
        
        while left<=right:
            mid = math.floor((right+left)/2)
            if (dp[mid]>=x):
                right = mid-1
            else:
                left = mid + 1
        return left
        
        
        
    def LIS(self , arr ):
        # write code here
        length = len(arr)
#         if length <=1: return 1#arr
#         min_arr = []
        lenn = [1]
        dp = [arr[0]]
#         dp = [1 for i in range(length)]
#         maxx = 0
        for i in range(1,length):
#             for j in range(i):
#                 if arr[i]>arr[j] and dp[i]<dp[j]+1:
#                     dp[i] = dp[j]+1
#                     maxx = maxx if maxx>dp[i] else dp[i]
            if arr[i] > dp[len(dp)-1]:
                dp.append(arr[i])
                lenn.append(len(dp))
            else:
                t = self.binary_S(arr[i],dp)
                dp[t] = arr[i]
                lenn.append(t+1)
                
        j=len(dp)
        res=[0 for _ in range(j)]
        
        for i in reversed(range(len(lenn))):
            if j>0:
                if lenn[i]==j:
                    j-=1
                    res[j] = arr[i]
            else: break
        return res
            
        
#         return max(self.LIS(arr[:,length-1])+1,self.LIS(arr[:,length-1]))
#         LIS()
# test = [2,1,5,3,6,4,8,9,7]
# return = [1,3,4,8,9]