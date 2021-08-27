#
# 
# @param arr int整型一维数组 the array
# @return int整型
# [2,3,4,5,5]
# 4 (continuous and no repeat)

# time complexity: O(n)
# space complexity:O(m) where m is the length of hashmap
class Solution:
    def maxLength(self , arr ):
        # write code here
        length = len(arr)
        if length<2: 
            return length
        
        # use hashmap to achieve O(1) on checking keys
        hashmap = {}
        biggest = 0
        left = -1
        
        for right in range(length):
            if arr[right] in hashmap:
                left = max(left,hashmap[arr[right]])
            biggest = max(biggest, right-left)
            hashmap[arr[right]] = right
            
        return biggest