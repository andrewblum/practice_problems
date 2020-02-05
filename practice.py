
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

# with memo 
def combinationSum(self, candidates: List[int], target: int, memo = {}) -> List[List[int]]:
    if target == 0: return [[]]
    results = []
    for each in candidates: 
        if target - each >= 0:
            if memo[target - each]:
              subs = memo[target - each]
            else: 
              subs = self.combinationSum(candidates, target - each)
              memo[target - each] = subs
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

def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    if (not p and q) or (not q and p): return False
    if not p and not q: return True
    if p.val != q.val: return False
    x = self.isSameTree(p.left, q.left)
    y = self.isSameTree(p.right, q.right)
    return x and y


# recursion without memo
def coinChange(self, coins, amount):
    if amount < 0: return -1
    if amount == 0: return 0
    results = []
    for coin in coins: 
        pos = self.helper(coins, amount - coin)
        if pos > -1:
            results.append(pos + 1)
    if results:
        return min(results)
    return -1

# recursion with memo
def coinChange2(self, coins: List[int], amount: int) -> int:
    memo = {}
    return self.helper(coins, amount, memo)
    
def helper(self, coins, amount, memo):
    if amount < 0: return -1
    if amount == 0: return 0
    results = []
    for coin in coins: 
        if not memo.get(amount - coin):
            memo[amount - coin] = self.helper(coins, amount - coin, memo)
        pos = memo[amount - coin]
        if pos > -1:
            results.append(pos + 1)
    if results:
        return min(results)
    return -1

# tabulation 

# [0, -1 OR 1, ]
# best we can do for each amount, in an array 
# look back COIN num of slots for each coin and add that + 1 for each coin and then compute the min of our choices for our current
# slot 

def coinChange(self, coins: List[int], amount: int) -> int:
    leastCoinsAtEachAmount = [-1] * (amount + 1)
    leastCoinsAtEachAmount[0] = 0
    for cur in range(1, amount + 1): 
        minPossibleChange = []
        for coin in coins: 
            if coin <= cur:
                used = leastCoinsAtEachAmount[cur - coin] + 1
                if used > 0:
                    minPossibleChange.append(used)
        if minPossibleChange: 
          leastCoinsAtEachAmount[cur] = min(minPossibleChange)
        else:
          leastCoinsAtEachAmount[cur] = -1
    return leastCoinsAtEachAmount[-1]

## BFS graph approach
def coinChange(self, coins: List[int], amount: int) -> int:
    if amount == 0: return 0
    q = [(amount, 1)]
    seen = set()
    while q: 
        cur = q.pop(0)
        for coin in coins:
            if cur[0] - coin == 0: return cur[1]
            if cur[0] - coin not in seen and cur[0] - coin > -1:
                q.append((cur[0] - coin, cur[1] + 1))
                seen.add(cur[0] - coin)
    return -1

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        colsToZero = set()
        rowsToZero = set()
        for rowIdx, row in enumerate(matrix):
            for colIdx, num in enumerate(row): 
                if num == 0: 
                    colsToZero.add(colIdx)
                    rowsToZero.add(rowIdx)
                    
        for col in colsToZero:
            self.writeZeroCol(col, matrix)
        for row in rowsToZero:
            self.writeZeroRow(row, matrix)
            
        
    def writeZeroRow(self, row, matrix):
        for col, _ in enumerate(matrix[row]):
            matrix[row][col] = 0
        
    def writeZeroCol(self, col, matrix):
        for row, _ in enumerate(matrix):
            matrix[row][col] = 0

#  the same thing but simpler if we make a copy of the 2d array first 

    # def setZeroes(self, matrix: List[List[int]]) -> None:
    #     """
    #     Do not return anything, modify matrix in-place instead.
    #     """
    #     copy = [row[:] for row in matrix]
    #     for rowIdx, row in enumerate(copy):
    #         for colIdx, num in enumerate(row): 
    #             if num == 0: 
    #                 self.writeZeroCol(colIdx, matrix)
    #                 self.writeZeroRow(rowIdx, matrix)
        
    # def writeZeroRow(self, row, matrix):
    #     for col, _ in enumerate(matrix[row]):
    #         matrix[row][col] = 0
        
    # def writeZeroCol(self, col, matrix):
    #     for row, _ in enumerate(matrix):
    #         matrix[row][col] = 0


class SolutionWordMaze:
    def exist(self, board: List[List[str]], word: str) -> bool:
        found = False
        first = word[0]
        rest = word[1:]
        for rowIdx, row in enumerate(board):
            for colIdx, col in enumerate(row): 
                if col == first:
                    seen = set([(rowIdx, colIdx)])
                    found = found or self.search(board, rest, (rowIdx, colIdx), seen)
        return found
            
    def search(self, board, word, pos, seen):
        if word == '': return True
        first = word[0]
        rest = word[1:]
        moves = self.getMoves(pos, board)
        for x, y in moves: 
            if board[x][y] == first and (x, y) not in seen:
                seen.add((x, y))
                self.search(board, rest, (x, y), seen.copy())

                
    def getMoves(self, pos, board):
        moves = []
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        for x, y in directions:
            newX = pos[0] + x
            newY = pos[1] + y
            tooLow = newX < 0 or newY < 0
            tooHigh = newX >= len(board) or newY >= len(board[0])
            if not tooLow and not tooHigh: 
                moves.append((newX, newY))
        return moves
                

# class Solution:
#     def exist(self, board: List[List[str]], word: str) -> bool:
#         for rowIdx, row in enumerate(board):
#             for colIdx, col in enumerate(row): 
#                 if col == word[0]:
#                     if self.search(board, word, (rowIdx, colIdx), set()):
#                         return True
#         return False
            
#     def search(self, board, word, pos, seen):
#         x, y = pos[0], pos[1]
#         if word == '': return True
#         if pos in seen: return False
#         if x < 0 or y < 0: return False 
#         if x >= len(board) or y >= len(board[0]): return False
#         if board[x][y] != word[0]: return False
        
#         seen.add(pos)
#         rest = word[1:]
#         found = (self.search(board, rest, (x + 1, y), seen) or 
#         self.search(board, rest, (x - 1, y), seen) or 
#         self.search(board, rest, (x, y + 1), seen) or 
#         self.search(board, rest, (x, y - 1), seen))
#         seen.remove(pos)
#         return found

def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
    results = []
    for rowIdx, row, in enumerate(matrix):
        for colIdx, col in enumerate(row): 
            if self.findEdge(matrix, rowIdx, colIdx):
                results.append([rowIdx, colIdx])
    return results
                
def findEdge(self, matrix, x, y): 
    q = [((x, y), set())]
    pacific, atlantic = False, False
    while q: 
        pos, seen = q.pop()
        xI, yI = pos
        for each in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            x, y = xI + each[0], yI + each[1]  
            if (x < 0 or y < 0): pacific = True
            if (x >= len(matrix) or y >= len(matrix[0])): atlantic = True
            if pacific and atlantic: return True
            if ((x >= 0 and y >= 0) and 
            (x < len(matrix) and y < len(matrix[0])) and 
            (x, y) not in seen):
                if matrix[xI][yI] >= matrix[x][y]:
                    seen = seen.copy()
                    seen.add((x, y))
                    q.append(((x, y), seen))
    return False


# pacific water flow with memo... 
# mark (x,y) with if we can get to the ocean from there 

# set the edges in the memo to already having pacific / atlantic 
# then only check for travel and P/A tag

def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
    results = []
    memo = {}
    # mark each edge as reaching the respective ocean to start the memo
    for rowIdx, row, in enumerate(matrix, start=1):
        for colIdx, col in enumerate(row, start=1): 
            if findEdges(matrix, rowIdx, colIdx, memo):
                results.append([rowIdx, colIdx])
    return results

def findEdge(self, matrix, x, y, memo):
    



# left or right subtree bigger
def treesolution(arr):
    # Type your solution here
    if not arr or len(arr) == 1: return ''
    arr.insert(0, None)
    left = treehelper(arr, 2)
    right = treehelper(arr, 3)
    if right > left: 
        return 'Right'
    elif right < left:
        return 'Left'
    else:
        return ''
    
def treehelper(arr, cur):
    if cur >= len(arr): return 0
    if arr[cur] == -1: return 0
    left = helper(arr, cur * 2)
    right = helper(arr, (cur * 2) + 1)
    return arr[cur] + left + right


class Node(object):
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

one = Node(1)
two = Node(2)
tree = Node(3)
four = Node(4)
five = Node(5)
six = Node(6)
seven = Node(7)

one.left = two 
one.right = three
two.left = four 
two.right = five
three.left = six
three.right = seven

def iterInorder(node):
    stack = []
    seen = set()
    while stack: 
        peek = stack[-1]
        if peek.left and peek.left not in seen: 
            stack.append(peek.left)
        if not peek.left or peek.left in seen: 
            print(peek.data)
            stack.pop()
            seen.add(peek)
            if peek.right: stack.append(peek.right)

iterInorder(one)


# longest palindrome 

def longest_palindrome(s):
  memo = {}
  if not s: return ""
  # for window in range(1, len(s) - 1):
  #   for startIdx, start in enumerate(s):
  #     endIdx = startIdx
  #     while (endIdx < len(s) - 1):
  #       print(startIdx, endIdx)
  #       endIdx += window
  if s[1] == s[-1]:
    r0 =  s[1] + longest_palindrome(s[1:-2]) + s[-1]
  r1 = longest_palindrome(s[1:])
  r2 = longest_palindrome(s[:-2])
  return max([r0, r1, r2])




def search_matrix(matrix, target)
    if matrix.length == 0: return False
    if matrix[0].length == 0: return False
    row = 0
    col = matrix[0].length - 1
    while (row >= 0 && row < matrix.length && col >= 0 && col < matrix[0].length)
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    return False


import heapq

class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.left = []
        self.right = []
        
    def addNum(self, num: int) -> None:
        if not self.left and not self.right: 
            heapq.heappush(self.right, num)
            return
        if not self.left or not self.right: 
            if self.left:
                heapq.heappush(self.right, num)
            else: 
                heapq.heappush(self.left, num * -1)
            if (self.left[0] * -1) > self.right[0]:
                n1 = heapq.heappop(self.right)
                n2 = heapq.heappop(self.left) * -1
                heapq.heappush(self.right, n2)
                heapq.heappush(self.left, n1 * -1)
            return
        
        # add num to the correct heap
        largest_left = self.left[0] * -1
        smallest_right = self.right[0]
        if num > smallest_right:
            heapq.heappush(self.right, num)
        elif num < largest_left:
            heapq.heappush(self.left, num * -1)
        else: 
            heapq.heappush(self.right, num)

        # rebalance the heaps       
        self.balance()
            
    def balance(self):
        if len(self.left) > len(self.right):
            n = heapq.heappop(self.left)
            heapq.heappush(self.right, n * -1)

        elif len(self.left) < len(self.right):
            n = heapq.heappop(self.right)
            heapq.heappush(self.left, n * -1)
        
    def findMedian(self) -> float:
        if len(self.left) == len(self.right):
            return (self.right[0] + (self.left[0] * -1)) / 2.0
        elif len(self.right) > len(self.left):
            return self.right[0]
        else: 
            return self.left[0] * -1



def findEdge(self, x, y, matrix, seen, prev):
    if x < 0 or y < 0: return 
    if x >= len(matrix) or y >= len(matrix[0]): return
    if matrix[x][y] < prev: return
    if (x, y) in seen: return
    seen.add((x, y))
    self.findEdge(x + 1, y, matrix, seen, matrix[x][y])
    self.findEdge(x - 1, y, matrix, seen, matrix[x][y])
    self.findEdge(x, y + 1, matrix, seen, matrix[x][y])
    self.findEdge(x, y - 1, matrix, seen, matrix[x][y])
    
def pacificAtlantic(self, matrix):
    if not matrix: return []
    p,a = set(),set()
    for rowIdx in range(len(matrix[0])):
        self.findEdge(0, rowIdx, matrix, p, -1)
        self.findEdge(len(matrix) -1, rowIdx, matrix, a, -1)
    for colIdx in range(len(matrix)):
        self.findEdge(colIdx, 0, matrix, p, -1)
        self.findEdge(colIdx, len(matrix[0]) - 1, matrix, a, -1)   
    return list(p&a)
                

def threeSum(self, nums: List[int]) -> List[List[int]]:
    if not nums: return []
    memo = collections.defaultdict(set)   
    for i, n in enumerate(nums):
        memo[n].add(i)
    results = set()
    for i1, n1 in enumerate(nums):
        for i2 in range(i1 + 1, len(nums)):
            m = memo.get((n1 + nums[i2]) * -1)
            if m and not (n1, nums[i2], (n1 + nums[i2]) * -1) in results:
                cur = set([i1, i2])
                left = m - cur
                if left:
                    sol = [n1, nums[i2], nums[left.pop()]]
                    sol.sort()
                    results.add(tuple(sol))
    return list(results)


# from collections import defaultdict 

# def shortest_substring(target, body): 
#     counts = defaultdict(lambda: 0)
#     start, end, count_of_alpha = 0, 0, 0 
#     min_start = 0
#     min_end = len(body) - 1
    
#     while end < len(body) :
#         if body[end] in target:
#             if counts[body[end]] == 0:
#                 count_of_alpha += 1
#             counts[body[end]] += 1
#         while count_of_alpha >= len(target):
#             if (end - start) < (min_end - min_start):
#                 min_start = start 
#                 min_end = end
#             counts[body[start]] -= 1
#             if counts[body[start]] == 0:
#                 count_of_alpha -= 1
#             start += 1
#         end += 1 
#     return body[min_start: min_end + 1]

    
    
    
# print(shortest_substring("abc", "aaccbca"))
# print(shortest_substring("abc", "aaccbc"))


# 1) 
# write more pseudo code 
# step to more detailed pseudo code 

# its a lot easier to refactor pseudo code than real code -- make it hapen then


# 2) 
# explain what youre doing and why youre doing it 

# 3)
# break out the small examples to target the problem or space of uncertintiy 



"""

 Input:
   array of positive numbers: [1, 2, 3, 4, 5]
   length of interval k: 3
 
 Output: [3,4,5]
 
 
 Input:
   arr: [5, 4, 3, 2, 1]
     k: 3

 Output: [5 4 3]
 
 Input:
   arr: [1, 2, 5, 2 ,1]
     k: 3
  
 Output: [2, 5, 2]


"""
[1, 1, 1, 1, 1]

[5, 4, 3, 2, 1]


max_so_far = 12 
cur_total = 12 
start = 0
end = 3


def max_sub_arr(arr, k):
  if len(arr) < k: 
    return arr
  max_so_far = sum(arr[:k])
  cur_total = max_so_far
  start, end = 0, k - 1
  for i in range(k, len(arr)): 
    cur_total = cur_total + arr[i] - arr[i - k]
    if cur_total > max_so_far:
      max_so_far = cur_total
      start = i - k + 1
      end = i 
  return arr[start:end + 1]


"""
 Input:
   array of positive numbers: [1, 2, 3, 4, 5, 6, 7]
   length of interval k: 2

 Output: [2,3], [4, 5], [6,7]
 

  k = 2
arr = [ 1 6 6 1 2 4  7  8  5 1 ]
      [           15 15 13 6 0 ]
                     <-- 
      [                         ]
                    -->
      
 
"""

"""
make two dynamic arrays comupting my best pair coming from the right side, and from the left side 

loop through our original array looking at one pair at a time , starting at the 2nd pair and ending before the last

we take our best options from the left and right of us and check them against a max
update max if needed 

repeat until the end 

return max 

"""


[ 1 6 6 1 2 4 7 8 5 1 ]

[ 0 0 13 ]

def max_3sub_arrays(arr, k): 
  max_from_left = [0] * k - 1
  max_from_right = [0] * k - 1
  max_so_far = 0
  
  for i in range(0, len(arr) - 1):
    max_so_far = max(max_so_far, arr[i] + arr[i + 1])
    max_from_left.append(max_so_far)
    
  i = len(arr) - 2
  max_so_far = 0
  while i >= 0:
    max_so_far = max(max_so_far, arr[i] + arr[i - 1])
    max_from_right.append(max_so_far)
    i -= 1
  max_from_right.reverse()



    
# more attention to detail 
# finding edge cases on basic verification 
# thinking through the impacts of each line of code added 

# always ask yourself "can i do better?"
# measure TWICE, cut once 

# think more carefully about the solutions you come up with
# can it be simpler, can it be cleaner

# pick better examples
# always check edge cases
# then pick smallest, significant example
# to check for typos / logic errors and so on


# YET MORE PSEUDO CODE 
# find the answers to your examples ahead of time


def isLongPressedName(name, typed):
  # gather all of a similar letter from the name, count how many we saw
  # gathr all of that letter from typed until consumed we hit a new letter, count how many we saw
  # if we didnt see at least count, fail 
  # if we get to the end of both strings, success


import collections
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
  # convert anagram to sorted count comparable object 
  # compare and group all similar 
  # put similar into lists
  if not strs: return []
  result = collections.defaultdict(list)
  for wd in strs:
      result[''.join(sorted(wd))].append(wd)
  return [x for x in result.values()]


def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    result = []
    i = 0
    intervals.sort(key=lambda x: x[0])
    while i < len(intervals): 
        start_of_interval = intervals[i][0]
        end_of_interval = intervals[i][1]
        while i < len(intervals) - 1 and end_of_interval >= intervals[i + 1][0]:
            i += 1
            end_of_interval = max(end_of_interval, intervals[i][1])
        result.append([start_of_interval, end_of_interval])
        i += 1
    return result

def orangesRotting(self, grid: List[List[int]]) -> int:
    for rI, row in enumerate(grid): 
        for cI, col in enumerate(row): 
            if col == 2:
                self.dfs(grid, rI, cI, 2)
    time = 2
    for row in grid: 
        for item in row: 
            time = max(time, item)
            if item == 1:
                return -1
    return time - 2
    
def dfs(self, grid, x, y, count):
    if x < 0 or y < 0: return 
    if x >= len(grid) or y >= len(grid[0]): return 
    if grid[x][y] == 0: return 
    if grid[x][y] >= count or grid[x][y] == 1: 
        grid[x][y] = count
        self.dfs(grid, x + 1, y, count + 1)
        self.dfs(grid, x - 1, y, count + 1)
        self.dfs(grid, x, y + 1, count + 1)
        self.dfs(grid, x, y - 1, count + 1)


def orangesRottingBFS(self, grid: List[List[int]]) -> int:
    rotten_oranges = deque()
    fresh_oranges = 0
    for rI, row in enumerate(grid): 
        for cI, cell in enumerate(row): 
            if cell == 2:
                rotten_oranges.append((rI, cI, 0))
            if cell == 1: 
                fresh_oranges += 1
    if fresh_oranges == 0: return 0      
    while rotten_oranges: 
        c_x, c_y, time = rotten_oranges.popleft()
        for x, y in [(c_x + 1, c_y), (c_x - 1, c_y), (c_x, c_y + 1), (c_x, c_y - 1)]:
            if x < 0 or y < 0: continue
            if x >= len(grid) or y >= len(grid[0]): continue
            if grid[x][y] != 1: continue
            grid[x][y] = 2
            fresh_oranges -= 1
            if fresh_oranges == 0: return time + 1
            rotten_oranges.append((x, y, time + 1))
    if fresh_oranges > 0: return -1
    return 0

class AllNodesKDistance:
    def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
        if not root: return []
        result = []
        if root.val == target.val:
            result += self.find_distance(root, 0, K)
        if root.right:
            setattr(root.right, 'p', root)
            result += self.distanceK(root.right, target, K)
        if root.left: 
            setattr(root.left, 'p', root)
            result += self.distanceK(root.left, target, K)
        return result
            
    def find_distance(self, root, count, limit):        
        if not root: return []
        if getattr(root, 'seen', None): return []
        if count == limit: return [root.val]
        if count > limit: return
        setattr(root, 'seen', True)
        result = []
        if root.right:
            result += self.find_distance(root.right, count + 1, limit)
        if root.left:
            result += self.find_distance(root.left, count + 1, limit)
        if getattr(root, 'p', None):
            result += self.find_distance(root.p, count + 1, limit)   
        return result 

        
class eventNode:
    def __init__(self, s, e):
        self.s = s
        self.e = e
        self.left = None
        self.right = None

class MyCalendar:

    def __init__(self):
        self.root = None

    def book(self, start: int, end: int) -> bool:
        if not self.root: 
            self.root = eventNode(start, end)
            return True
    
        to_visit = [self.root]
        while to_visit: 
            cur = to_visit.pop()
            if start >= cur.e:
                if cur.right:
                    to_visit.append(cur.right)
                else: 
                    cur.right = eventNode(start, end)
                    return True
            elif end <= cur.s:
                if cur.left:
                    to_visit.append(cur.left)
                else: 
                    cur.left = eventNode(start, end)
                    return True
            else: 
                return False
    
def titleToNumber(self, s: str) -> int:
    if not s: return
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alpha = {l:i + 1 for i, l in enumerate(alphabet)}
    total = 0
    i = len(s) - 1
    for l in s: 
        total += int(alpha[l]) * (26 ** i)
        i -= 1
    return total

def isBipartite(self, graph: List[List[int]]) -> bool:
    colors = {}

    def can_be_colored(node):
        for child in graph[node]:
            if child in colors:
                if colors[child] == colors[node]:
                    return False
            else:
                colors[child] = colors[node] * -1
                if not can_be_colored(child):
                    return False
        return True
    
    for node, _ in enumerate(graph): 
        if node not in colors: 
            colors[node] = 1
            if not can_be_colored(node):
                return False 
    return True

def gameOfLife(board: List[List[int]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    result = [board[i][:] for i, _ in enumerate(board)]
    for x, row in enumerate(board):
        for y, cell in enumerate(row):
            result[x][y] = helper(x, y, board, cell)

    for x, row in enumerate(board):
        for y, cell in enumerate(row):
            board[x][y] = result[x][y]
    
def helper(oldx, oldy, board, cell):
    neighbors = 0
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            if x == 0 and y == 0: continue
            newx = oldx + x
            newy = oldy + y
            if newx < 0 or newy < 0: continue
            if newx >= len(board) or newy >= len(board[0]): continue
            if board[newx][newy] == 1:
                neighbors += 1
    
    if neighbors == 3: 
        return 1
    if cell == 1 and neighbors == 2:
        return 1
    return 0

def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:  
    if not t2 and not t1: return
    if t2 and t1:
        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        return t1
    else:
        return t1 or t2

def invertBT(n):
    if root:
        n.right, n.left = invertBT(n.left), invertBT(n.right)
        return root