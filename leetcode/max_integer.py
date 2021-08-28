#
# 最大数
# @param nums int整型一维数组 
# @return string字符串

# [2,20,23,4,8]
# "8423220"
import heapq
class Solution:
    def fill(self, s, l):
        if len(s)==l:return s
        t = s[-1]
        while len(s)<l:
            s+=t
        return s
    
    def solve(self , nums ):
        # write code here
        if len(nums)==0:return ''
        if len(nums)==1:return str(nums[0])
        num = []
        maxlen = 0
        maxitem = 0
        for i in range(len(nums)):
            if nums[i]>maxitem:
                maxitem = nums[i]
            tmp = str(nums[i])
            l = len(tmp)
            num.append(tmp)
            if maxlen<l:
                maxlen=l
                
        if maxitem==0:return '0'
        hashmap = dict()
        for i in range(len(num)):
            prev = num[i]
            num[i] = self.fill(num[i],maxlen)
            if num[i] in hashmap:
                hashmap[num[i]].append(int('-'+prev))
            else:
                hashmap[num[i]]=[int('-'+prev)]
                    
        res = ''
        num = sorted(num,reverse=True)
        for s in num:
            j = heapq.heappop(hashmap[s])
            res+=str(-j)
        return res
        
        
        