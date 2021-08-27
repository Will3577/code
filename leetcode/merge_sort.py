#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# 将给定数组排序
# @param arr int整型一维数组 待排序的数组
# @return int整型一维数组
#
class Solution:

    def MySort(self , arr ): #merge sort
        # write code here
        length = len(arr)
        if length>1:
            mid = length//2
            L = arr[:mid]
            R = arr[mid:]
            
            L = self.MySort(L)
            R = self.MySort(R)
            #
            i=0
            j=0
            tmp=[]
            while i<len(L) and j<len(R):
                if L[i]>=R[j]:
                    tmp.append(R[j])
                    j+=1
                else:
                    tmp.append(L[i])
                    i+=1
            
            # fill array
            while i<len(L):
                tmp.append(L[i])
                i+=1
            
            while j<len(R):
                tmp.append(R[j])
                j+=1
                
            return tmp
        return arr
                    
            
            
            
            
            
            
            
            
            
            
        