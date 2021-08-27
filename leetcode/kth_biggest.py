# -*- coding:utf-8 -*-
# [10,10,9,9,8,7,5,6,4,3,4,2],12,3
# 9
# time complexity: O(n)
# space complexity: O(n)
class Solution:
    
    def partition(self, arr, low, high):
        i = (low-1)         # index of smaller element
        pivot = arr[high]     # pivot

        for j in range(low, high):

            # If current element is smaller than or
            # equal to pivot
            if arr[j] > pivot:
                # increment index of smaller element
                i = i+1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i+1], arr[high] = arr[high], arr[i+1]
        return (i+1)
    
    def qSort(self, arr, low, high, K):
        p = self.partition(arr,low,high)
        
        if K==p-low+1: # if p is Kth biggest
            return arr[p]
        elif p-low+1>K: # left side
            return self.qSort(arr,low,p-1,K)
        else:
            return self.qSort(arr,p+1,high,K-(p-low+1))

    
    def findKth(self, arr, n, K):        
        return self.qSort(arr,0,n-1,K)
    