#
# 
# @param num int整型一维数组 
# @return int整型二维数组

#[-10,0,10,20,-10,-40]
#[[-10,-10,20],[-10,0,10]]

class Solution:
    def threeSum(self , num ):
        # write code here
        if len(num)<3:
            return []
        
        sort = num
        sort.sort()
        res = set()
        ress = []
        length = len(num)
        for i in range(length):
            for j in range(i+1,length):
                if sort[i]+sort[j]>0: 
                    break
                if sort[i]+sort[j]<=0:
                    for k in range(j+1,length):
                        if sort[i]+sort[j]+sort[k]>0:
                            break
                        if sort[i]+sort[j]+sort[k]==0:
                            tmp = (sort[i],sort[j],sort[k])
                            tmpp = [sort[i],sort[j],sort[k]]
#                             if len(res)==0 or res[-1]!=tmp:
#                                 res.append(tmp)
                            old = len(res)
                            res.add(tmp)
                            if old < len(res):
                                ress.append(tmpp)
        return ress
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        