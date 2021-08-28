class Interval:
    def __init__(self, a=0, b=0):
        self.start = a
        self.end = b

#
# @param intervals Interval类一维数组 
# @return Interval类一维数组
# [[10,30],[20,60],[80,100],[150,180]]
# [[10,60],[80,100],[150,180]]

class Solution:
    def merge(self , intervals ):
        # write code here
        if len(intervals)<2: return intervals
        tmp_list = []
        for i in range(len(intervals)):
            tmp_list.append([intervals[i].start,
                             intervals[i].end])
        sorted_list = sorted(tmp_list)
        res=[]
        cur = Interval(sorted_list[0][0],sorted_list[0][1])
        for i in range(1,len(sorted_list)):
            interval = Interval(sorted_list[i][0],sorted_list[i][1])
            if cur.end>=interval.start and cur.start<=interval.start:
                cur.end = max(cur.end,interval.end)
                if i==len(intervals)-1:
                    res.append(cur)
            elif i!=len(intervals)-1:
                res.append(cur)
                cur = interval
            else:
                res.append(cur)
                res.append(interval)

        return res






