# -*- coding:utf-8 -*-
# "abc1234321ab",12
# 7
class Solution:
    def getLongestPalindrome(self, A, n):
        # write code here
        hashmap = dict()
        for i in range(n):
            if A[i] in hashmap:
                hashmap[A[i]].append(i)
            else:
                hashmap[A[i]] = [i]
        res = 1
        for i in range(n):
            for j in reversed(hashmap[A[i]]):
                if j<=i:break
                tmp = 2

                for k in range(1,(j-i)):
                    if i+k>j-k or A[i+k]!=A[j-k]:
                        break
                    elif i+k==j-k:
                        tmp+=1
                        break
                    elif A[i+k]==A[j-k]:
                        if i==3:
                        tmp+=2

            # 3 9/4 8/5 7/6
            # 3 8/4 7/5 6
                if tmp>res and (i+k==j-k or A[i+k]==A[j-k]):
                    res=tmp
        return res
        
        
        
        
        
        
        
        
        