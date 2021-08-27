# -*- coding:utf-8 -*-
# [0,1,2,1,2],3
# [0,1,1]
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if k==0: return []
        res = tinput[:k]
        res.sort()
        
        for i in range(k,len(tinput)):
            # update res if tinput[i]<res[-1]
        
            if tinput[i]<res[-1]:
                res[-1] = tinput[i]
                for j in reversed(range(1,k)):
                    if res[j]<res[j-1]:
                        tmp=res[j-1]
                        res[j-1]=res[j]
                        res[j]=tmp
                    else:
                        break
                    
        return res
        
        
        
        
        
        
        
        
        
        
        