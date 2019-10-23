
def findMultiKids(state, deadends):
    vals = []
    up = str(int(state[-1]) + 1)
    down = str(int(state[-1]) - 1)
    none = state[-1]

    if (up == '10'): up = '0'
    if (down == '-1'): down = '9'

    if len(state) == 1: 
      return [up, down, none]
    result = findKids(state[:-1], deadends)
    for each in result: 
        if each + up not in deadends: vals.append(each + up)
        if each + down not in deadends: vals.append(each + down)
        if each + none not in deadends: vals.append(each + none)
    return vals

class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        if '0000' in deadends: return -1
        seen = set()
        q = ['0000']
        depth = {'0000': 0}
        dead = set(deadends)
        while q:
            cur = q.pop(0)
            if cur == target: return depth[cur] - 1
            children = findKids(cur, dead)
            for child in children: 
                if child not in seen: 
                    seen.add(child)
                    q.append(child)
                    depth[child] = depth[cur] + 1
        return -1
    
def findKids(state, deadends):
    vals = {}
    for i, each in enumerate(state): 
        up = str(int(each) + 1)
        down = str(int(each) - 1)
        if (up == '10'): up = '0'
        if (down == '-1'): down = '9'
        copy = state[:]
        if copy not in deadends: vals[copy] = ''
        copy = copy[:i] + up + copy[i + 1:]
        if copy not in deadends: vals[copy] = ''
        copy = copy[:i] + down + copy[i + 1:]
        if copy not in deadends: vals[copy] = ''
    return list(vals)


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        if target == 0: return [[]]
        results = []
        for each in candidates: 
            if target - each >= 0:
                subs = self.combinationSum(candidates, target - each)
                for lst in subs: 
                    lst.append(each)
                results = results + subs
        gross = set([tuple(sorted(x)) for x in results])
        return [list(x) for x in set(gross)]

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        grid = [[1 for x in range(m)]] * n
        for row in range(1, n):
            for col in range(1, m):
                grid[row][col] = grid[row - 1][col] + grid[row][col - 1]
        return grid[-1][-1]
