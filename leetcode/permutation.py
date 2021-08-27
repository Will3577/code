# -*- coding:utf-8 -*-
class Solution:
    '''
    "abc"
    ["abc","acb","bac","bca","cab","cba"]
    '''
    def swap(self, ss, pos1, pos2):
        ss = list(ss)
        ss[pos1],ss[pos2] = ss[pos2],ss[pos1]
        return ''.join(ss)
    
    def perm(self, pos, ss, hashmap):
        
        if pos+1==len(ss):
#             hashmap[ss] = ss
            hashmap.update({ss:ss})
            return 
        
        for i in range(pos,len(ss)):
            
            ss = self.swap(ss,pos,i)
            self.perm(pos+1,ss,hashmap)
            ss = self.swap(ss,pos,i)
            
        return 
    
    def Permutation(self, ss):
        # write code here
        if len(ss)==0: return []
        hashmap = dict()
        
        self.perm(0,ss,hashmap)
        res = list(hashmap.keys())
        res.sort()
        return res
        
        
        
        
        
        
        
        
        
        