
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

def diameterOfBinaryTree(self, root: TreeNode) -> int:
    self.ans = 0 
    def max_depth(n):
        if not n: return 0
        right = max_depth(n.right)
        left = max_depth(n.left)
        # if i know the max depths of the paths to the left and right of me
        # then the diameter is those added, or my currrent biggest
        self.ans = max(self.ans, left + right)
        # if i have the longest depths of the left and right
        # then my answer it the bigger path + 1 to account for the node i am on
        return 1 + max(left, right)
    max_depth(root)
    return self.ans

def minPathSum(self, grid: List[List[int]]) -> int:
    for x, cur in enumerate(grid[0]):
        if x > 0: 
            grid[0][x] = grid[0][x - 1] + cur
    for y, cur in enumerate(grid):
        if y > 0: 
            grid[y][0] = grid[y - 1][0] + grid[y][0]
    
    
    for x, row in enumerate(grid):
        for y, _ in enumerate(row):
            if x > 0 and y > 0:
                grid[x][y] = min(grid[x - 1][y], grid[x][y - 1]) + grid[x][y]
        
    return grid[-1][-1]

def minCostToMoveChips(self, chips: List[int]) -> int:
    odds = sum(x % 2 for x in chips)
    return min(odds, len(chips) - odds)

def findWords(self, words: List[str]) -> List[str]:
    def sr(word):
        r1 = set(['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'])
        r2 = set(['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'])
        r3 = set(['z', 'x', 'c', 'v', 'b', 'n', 'm'])
        s = r3
        if word[0].lower() in r1: 
            s = r1
        elif word[0].lower() in r2: 
            s = r2
        for i, l in enumerate(word):
            if l.lower() not in s: 
                return False
        return True
    
    result = []
    for word in words:
        if sr(word): result.append(word)
    return result 

def numberOfSteps (self, num: int) -> int:
    steps = 0
    while num > 0:
        if num % 2 == 0: 
            num = num / 2
        else: 
            num -= 1
        steps += 1
    return steps

def second_largest(node):
    while node.right: 
        if not node.right.right: 
            parent = node
        node = node.right 

    if node.left:
        node = node.left 
        while node.right:
            node = node.right
    else: 
        node = parent
    return parent
    
 def find_repeat(numbers):
    floor = 1
    ceiling = len(numbers) - 1
    while floor < ceiling:
        # Divide our range 1..n into an upper range and lower range
        # (such that they don't overlap)
        # Lower range is floor..midpoint
        # Upper range is midpoint+1..ceiling
        midpoint = floor + ((ceiling - floor) // 2)
        lower_range_floor, lower_range_ceiling = floor, midpoint
        upper_range_floor, upper_range_ceiling = midpoint+1, ceiling
        # Count number of items in lower range
        items_in_lower_range = 0
        for item in numbers:
            # Is it in the lower range?
            if item >= lower_range_floor and item <= lower_range_ceiling:
                items_in_lower_range += 1
        distinct_possible_integers_in_lower_range = (
            lower_range_ceiling
            - lower_range_floor
            + 1
        )
        if items_in_lower_range > distinct_possible_integers_in_lower_range:
            # There must be a duplicate in the lower range
            # so use the same approach iteratively on that range
            floor, ceiling = lower_range_floor, lower_range_ceiling
        else:
            # There must be a duplicate in the upper range
            # so use the same approach iteratively on that range
            floor, ceiling = upper_range_floor, upper_range_ceiling
    # Floor and ceiling have converged
    # We found a number that repeats!
    return floor

def find_duplicate(int_list):
    n = len(int_list) - 1
    # STEP 1: GET INSIDE A CYCLE
    # Start at position n+1 and walk n steps to
    # find a position guaranteed to be in a cycle
    position_in_cycle = n + 1
    for _ in range(n):
        position_in_cycle = int_list[position_in_cycle - 1]
        # we subtract 1 from the current position to step ahead:
        # the 2nd *position* in a list is *index* 1
    # STEP 2: FIND THE LENGTH OF THE CYCLE
    # Find the length of the cycle by remembering a position in the cycle
    # and counting the steps it takes to get back to that position
    remembered_position_in_cycle = position_in_cycle
    current_position_in_cycle = int_list[position_in_cycle - 1]  # 1 step ahead
    cycle_step_count = 1
    while current_position_in_cycle != remembered_position_in_cycle:
        current_position_in_cycle = int_list[current_position_in_cycle - 1]
        cycle_step_count += 1
    # STEP 3: FIND THE FIRST NODE OF THE CYCLE
    # Start two pointers
    #   (1) at position n+1
    #   (2) ahead of position n+1 as many steps as the cycle's length
    pointer_start = n + 1
    pointer_ahead = n + 1
    for _ in range(cycle_step_count):
        pointer_ahead = int_list[pointer_ahead - 1]
    # Advance until the pointers are in the same position
    # which is the first node in the cycle
    while pointer_start != pointer_ahead:
        pointer_start = int_list[pointer_start - 1]
        pointer_ahead = int_list[pointer_ahead - 1]
    # Since there are multiple values pointing to the first node
    # in the cycle, its position is a duplicate in our list
    return pointer_start

def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return
        adjLst = []
        newnodes = {}
        q = [node]
        while q: 
            cur = q.pop()
            if not newnodes.get(cur.val, False):
                newnodes[cur.val] = Node(cur.val)
                for n in cur.neighbors: 
                    adjLst.append([cur.val, n.val])
                    q.append(n)
        
        for adj in adjLst:
            newnodes[adj[0]].neighbors.append(newnodes[adj[1]])
            
        return newnodes[node.val]

def cloneGraph2(self, node: 'Node') -> 'Node':
        if not node: return
        newnodes = {}
        newnodes[node.val] = Node(node.val)
        q = [node]
        while q: 
            cur = q.pop()
            for n in cur.neighbors: 
                if n.val not in newnodes:
                    newnodes[n.val] = Node(n.val)
                    q.append(n)
                newnodes[cur.val].neighbors.append(newnodes[n.val]) 
        return newnodes[node.val]

from collections import defaultdict
def deepestLeavesSum(self, root: TreeNode) -> int:
    if not root: return 0
    max_depth =  0
    depths = defaultdict(list)
    q = [(root, 0)]
    while q: 
        cur, depth = q.pop()
        if cur: 
            if depth > max_depth: max_depth = depth
            depths[depth].append(cur)
            q.append((cur.left, depth + 1))
            q.append((cur.right, depth + 1))
    
    return sum([node.val for node in depths[max_depth]])

def sumEvenGrandparent(self, root: TreeNode) -> int:
    def adder(node):
        total = 0
        if node.right: 
            if node.right.right:
                total += node.right.right.val
            if node.right.left:
                total += node.right.left.val
        if node.left:
            if node.left.right:
                total += node.left.right.val
            if node.left.left:
                total += node.left.left.val
        return total 
    
    q = [root]
    total = 0
    while q:
        cur = q.pop() 
        if cur:
            if cur.val % 2 == 0: 
                total += adder(cur)
            q.append(cur.left)
            q.append(cur.right)
    return total 

def queensAttacktheKing(self, queens: List[List[int]], king: List[int]) -> List[List[int]]:
    result = []
            
    def search(startx, starty, movex, movey):
        if startx < 0 or starty < 0: return
        if startx > 7 or starty > 7: return
        if [startx, starty] in queens:
            result.append([startx, starty])
            return
        search(startx + movex, starty + movey, movex, movey)
    
    for x in [1, 0, -1]:
        for y in [1, 0, -1]:
            if x == 0 and y == 0: continue
            search(king[0], king[1], x, y)
    
    return result
    


def dfs(queen,diag1,diag2,col):
    
    if len(queen)==n:
        res.append(queen+[])
        return
    i=len(queen)
    
    for j in range(n):
        if (i+j in diag1) or (i-j in diag2) or (j in col):
            continue
        else:
            diag1.add(i+j)
            diag2.add(i-j)
            col.add(j)
            queen.append(j)
            
            dfs(queen,diag1,diag2,col)
            
            queen.pop()
            col.remove(j)
            diag2.remove(i-j)
            diag1.remove(i+j)
            
res=[]
dfs([],set(),set(),set())
return [  ["."*j+'Q'+ (n-j-1)*'.' for j in ans] for ans in res]

def nqueens(n):
    def checkDiag(board, x, y):
        for rowi, row in enumerate(board):
            if board[rowi][y] == 'Q':
                return False
        return True


    def checkCol(board, x, y):          
        for x2 in [1, -1]:
            for y2 in [1, -1]:
                nx = x + x2
                ny = y + y2
                while nx >= 0 and ny >= 0 and nx < n and ny < n:
                    if board[nx][ny] == 'Q': 
                        return False 
                    nx += x2
                    ny += y2
        return True


    def helper(board, queens, srow):
        if queens == n:
            solutions.append(board)
            return
        for rowi in range(srow, len(board)):
            row = board[rowi]
            for coli, col in enumerate(row):
                if board[rowi][coli] == '.':
                    if checkDiag(board, rowi, coli) and checkCol(board, rowi, coli):
                        if row.count('Q') == 0:
                            copy = board[:]
                            copy[rowi] = copy[rowi][:coli] + 'Q' + copy[rowi][coli+1:]
                            helper(copy, queens + 1, rowi + 1)

    solutions = []
    board = ["." * n for _ in range(n)]
    helper(board, 0, 0) 
    return solutions 

def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
    self.max_path = 0
    self.memo = {}
    
    def dfs(x, y, prev):

        if x < 0 or y < 0: 
            return 0
        
        if x >= len(matrix) or y >= len(matrix[0]): 
            return 0
        
        if prev >= matrix[x][y]: 
            return 0
            
        if (x, y) in self.memo:
            return self.memo[(x,y)]
        
        a = dfs(x+1, y, matrix[x][y])
        b = dfs(x-1, y, matrix[x][y])
        c = dfs(x, y+1, matrix[x][y])
        d = dfs(x, y-1, matrix[x][y])
        
        path = max(a,b,c,d) + 1
        self.memo[(x,y)] = path
        self.max_path = max(path, self.max_path) 
        return path

    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            dfs(x, y, -100000)
    return self.max_path
    
def removeOuterParentheses(self, S: str) -> str:
    ans = ''
    count = 0 
    for i in range(len(S)):
        if S[i] == ')':
            count -= 1
        if count != 0:
            ans += S[i]
        if S[i] == '(':
            count += 1
    return ans


def productExceptSelf(self, nums: List[int]) -> List[int]:
    if len(nums) == 1: return nums
    prods = nums[:]
    prod_so_far = 1
    for i in range(len(nums)):
        prods[i] = prod_so_far
        prod_so_far *= nums[i]
    prod_so_far = 1
    for i in range(len(nums) - 1, -1, -1):
        prods[i] *= prod_so_far 
        prod_so_far *= nums[i] 
    return prods


def superBal(node):
    return abs(maxD(node) - minD(node)) < 2

def maxD(node):
    if not node: 
        return 0
    left = maxD(node.left)
    right = maxD(node.right)
    return max(left, right) + 1

def minD(node):
    if not node: return 0
    left = minD(node.left)
    right = minD(node.right)
    if left == 0: 
        return right + 1
    if right == 0:
        return left + 1
    return min(left, right) + 1

def superBal2(node):
    first = True
    minD, maxD = 0, 0 
    q = [(node, 1)]
    while q: 
        cur, lvl = q.pop()
        if cur: 
            if not cur.left and not cur.right and first:
                first = False
                minD = lvl
                maxD = lvl
            if not cur.left and not cur.right and first:
                minD = min(lvl, minD)
                maxD = max(lvl, maxD) 
            q.append((cur.right, lvl + 1))
            q.append((cur.left, lvl + 1))
    print(minD, maxD)
    return abs(maxD - minD) < 2

def get_permutations(string):
    # Generate all permutations of the input string
    if len(string) <= 1: 
        return set([string])
    ans = set()
    cur = string[0]
    string = string[1:]
    for s in get_permutations(string):
        for i in range(len(s) + 1):
            n = s[:i] + cur + s[i:]
            ans.add(n)

    return ans

def change_possibilities(amount, denominations):

    # Calculate the number of ways to make change
    
    def helper(amount, denominations, memo):
        if amount == 0: return 1
        if amount < 0: return 0
        if not denominations: return 0
        ways = 0
        c = denominations[0] 
        while amount >= 0:
                if amount in memo: 
                    ways += memo[amount]
                memo[amount] = change_possibilities(amount, denominations[1:])
                ways += memo[amount]
                amount -= c
        return ways
    
    m = {}
    return helper(amount, denominations, m)

def change_possibilities2(amount, denominations):

    # Calculate the number of ways to make change
    
    m = [0] * (amount + 1)
    m[0] = 1
    
    for c in denominations: 
        for amt in range(c, amount + 1):
            m[amt] += ( m[amt - c] )
            print(m)
    return m[-1]

def sortByBits(self, arr: List[int]) -> List[int]:
    bits = defaultdict(list)
    for n in arr:
        ones = bin(n).count('1')
        bits[ones].append(n)
    ans = []
    for k in sorted(bits.keys()):
        ans += sorted(bits[k])
    return ans

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

class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        self.t = []
        def trav(node):
            if not node: return 
            for c in node.children:
                trav(c)
            self.t.append(node.val)
        trav(root)
        return self.t

def searchBST(self, root: TreeNode, val: int) -> TreeNode:
    if not root: return None
    if val > root.val:
        return self.searchBST(root.right, val)
    if val < root.val:
        return self.searchBST(root.left, val)
    return root

def maxSubArray(self, nums: List[int]) -> int:
    for i in range(1, len(nums)):
        nums[i] = max(nums[i], nums[i - 1] + nums[i])
    return max(nums)

def matrixScore(self, A: List[List[int]]) -> int:
    for i in range(len(A)): 
        if A[i][0] == 0: 
            A[i] = [int(not n) for n in A[i]]

    for col in range(1, len(A[0])):
        zeros = 0
        for row in range(len(A)): 
            if A[row][col] == 0: 
                zeros += 1
                
        if zeros > len(A) - zeros:
            for row in range(len(A)): 
                A[row][col] = int(not A[row][col])

    s = 0
    for row in A: 
        s += int(''.join([str(n) for n in row]), base=2)
    return s

def minWindow(self, s: str, t: str) -> str:
        min_window = None
        counts = {} 
        for l in t: 
            counts[l] = counts.get(l, 0) + 1
        q = deque()
        missing = set(t)
        for i, l in enumerate(s): 
            if l in counts: 
                q.append((l, i))
                counts[l] -= 1
                if counts[l] == 0 and l in missing:
                    missing.remove(l)
                        
            while not missing:
                if not min_window or ((q[-1][1] + 1) - q[0][1]) < len(min_window): 
                    min_window = s[q[0][1]:q[-1][1] + 1]
                removed_letter, idx = q.popleft()
                counts[removed_letter] += 1
                if counts[removed_letter] > 0:
                    missing.add(removed_letter)
                
        if not min_window:
            return ''
        return min_window

class Node: 
    def __init__(self, val):
        self.val = val
        self.children = {}
        self.is_word = False

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node('')
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        current = self.root
        for i, letter in enumerate(word): 
            if current.children.get(letter):
                current = current.children.get(letter)
            else:
                current.children[letter] = Node(letter)
                current = current.children[letter]
            if i == len(word) - 1:
                current.is_word = True
            

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        current = self.root
        for letter in word: 
            current = current.children.get(letter)
            if not current:
                return False
        return current.is_word
            
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        current = self.root
        for letter in prefix: 
            current = current.children.get(letter)
            if not current:
                return False
        return True


class Node: 
    def __init__(self, val):
        self.val = val
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = Node('')
        
    def insert(self, word: str) -> None:
        current = self.root
        for i, letter in enumerate(word): 
            if current.children.get(letter):
                current = current.children.get(letter)
            else:
                current.children[letter] = Node(letter)
                current = current.children[letter]
            if i == len(word) - 1:
                current.is_word = True


class StreamChecker:

    def __init__(self, words: List[str]):
        self.pointers = []
        self.trie = Trie()
        for word in words:
            self.trie.insert(word)

            
    def query(self, letter: str) -> bool:
        # traverse pointers down trie by the letter
        # remove pointers without the letter child       
        new = []
        for pointer in self.pointers + [self.trie.root]:
            if letter in pointer.children:
                new.append(pointer.children[letter])
        self.pointers = new
            
        # see if any of the pointers are now on a word 
        return any(pointer.is_word for pointer in self.pointers)

import random

def get_random(floor, ceiling):
    return random.randrange(floor, ceiling + 1)

def shuffle(the_list):
    for i in range(len(the_list)):
        selection_idx = get_random(i, len(the_list) - 1)
        the_list[i], the_list[selection_idx] = the_list[selection_idx], the_list[i]
        # selected_num = the_list.pop(selection_idx)
        # the_list.insert(0, selected_num)

def rand5():
    return random.randint(1, 5)


def rand7():

    results = [
        [1, 2, 3, 4, 5],
        [6, 7, 1, 2, 3],
        [4, 5, 6, 7, 1],
        [2, 3, 4, 5, 6],
        [7, 0, 0, 0, 0],
    ]
    
    row = 4
    col = 4
    while row >= 4 and col > 0:
        row = rand5() - 1
        col = rand5() - 1
    return results[row][col]

def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
    graph, dist, prev = {}, {}, {}
    heap = []
    # build our graph into a dictonary   
    for edge in times: 
        source = edge[0]
        target = edge[1]
        weight = edge[2]
        node = graph.get(source, [])
        node.append((target, weight))
        graph[source] = node
    # setup for dijkstras
    for node in graph.keys():
        dist[node] = float('inf')
        prev[node] = None
        heapq.heappush(heap, node)
        if node == K:
            dist[node] = 0
    
    while heap: 
        cur = heapq.heappop(heap)
        for neighbor in graph[cur]:
            alt_dis = dist[cur] + neighbor[1]
            if alt_dis < dist.get(neighbor[0], 0): 
                dist[neighbor[0]] = alt_dis
                prev[neighbor[0]] = cur
    
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    def explore(x, y, grid): 
        if (x < 0 or y < 0 or 
        x >= len(grid) or y >= len(grid[0]) or 
        grid[x][y] != 1): 
            return 0 
        grid[x][y] = 'x'
        area = 0
        area += explore(x - 1, y, grid) 
        area += explore(x, y - 1, grid) 
        area += explore(x + 1, y, grid) 
        area += explore(x, y + 1, grid) 
        return area + 1
    
    max_area = 0
    for x, row in enumerate(grid):
        for y, space in enumerate(row): 
            if space == 1:
                max_area = max(explore(x, y, grid), max_area)
    return max_area

def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
    if not s or not t:
        return False
    
    def check_trees(t):
        if not t: return ''
        return str(t.val) + check_trees(t.left) + check_trees(t.right)
        
    if s.val == t.val: 
        if check_trees(t) == check_trees(s):
            return True
    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)


def suggestedProducts_slow(products: List[str], searchWord: str) -> List[List[str]]:
    products = sorted(products)
    result = []
    for i in range(len(searchWord)): 
        sub_result = []
        prefix = searchWord[0:i+1]
        for word in products: 
            if len(sub_result) == 3: 
                break
            if word.startswith(prefix):
                sub_result.append(word)
        result.append(sub_result)
    return result

#   to improve store words in a trie 

#   could also binary search for a starting point in the products list instead of
#   looping through the whole thing 
            
# 946 
def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
    stack = []
    for p in pushed: 
        stack.append(p)
        while stack[-1] == popped[0]:
            popped.pop(0)
            stack.pop()
            if not stack:
                break
    return not stack and not popped

def findPairs(self, nums: List[int], k: int) -> int:
    if k < 0: 
        return 0
    result = set()
    dic = {}
    for i, x in enumerate(nums): 
        dic[x] = dic.get(x, []) + [i]
        
    for i, n in enumerate(nums):
        if (n - k) in dic and (len(dic[n-k]) > 1 or dic[n-k][0] != i):
            result.add(tuple(sorted([n, n-k])))
        elif (n + k) in dic and (len(dic[n+k]) > 1 or dic[n+k][0] != i): 
            result.add(tuple(sorted([n, n+k])))
            
    return len(result)

class QuadTree:
    def construct(self, grid: List[List[int]]) -> 'Node':
        return self.helper(0, 0, len(grid), len(grid[0]), grid)
 
    
    def helper(self, start_x, start_y, end_x, end_y, grid):
        root = Node(True, True, None, None, None, None)
        if start_x >= end_x or start_y >= end_y: 
            return None
        
        if not self.all_same(start_x, start_y, end_x, end_y, grid):
            root.isLeaf = False
            # divide into 4 parts 
            half = len(grid) // 2
            root.topLeft = self.helper(start_x, start_y, end_x//2, end_y//2, grid)
            root.topRight = self.helper(start_x, start_y + half, end_x//2, end_y, grid)
            root.bottomLeft = self.helper(start_x + half, start_y, end_x, end_y//2, grid)
            root.bottomRight = self.helper(start_x + half, start_y + half, end_x, end_y, grid)
        else: 
            root.val = grid[start_x][start_y]
        return root
    
    def all_same(self, start_x, start_y, end_x, end_y, grid):
        all_same = grid[start_x][start_y]
        for x in range(start_x, end_x): 
            for y in range(start_y, end_y):
                if grid[x][y] != all_same:
                    return False
        return True


def singleNumber(self, nums: List[int]) -> int:
    n = nums[0]
    for i in range(1, len(nums)): 
        n ^= nums[i]
    return n 

def closedIsland(self, grid: List[List[int]]) -> int:
    def explore(x, y, grid):
        # we go went of bounds, so can't be surrounded by water         
        if (x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0])): 
            return False 
        # we hit water which is fine but a dead end
        if grid[x][y] == 1 or grid[x][y] == 'x':
            return True
        # mark as seen and continue exploring island   
        grid[x][y] = 'x'
        up = explore(x - 1, y, grid) 
        left = explore(x, y - 1, grid) 
        down = explore(x + 1, y, grid) 
        right = explore(x, y + 1, grid) 
        return up and down and left and right
            
    closed_islands = 0
    for x, row in enumerate(grid):
        for y, space in enumerate(row):
            if space == 0:
                closed_islands += explore(x, y, grid)
    return closed_islands

# floyd's cycle detection! 
def isHappy(self, n: int) -> bool:
    next_num = lambda n: sum([int(x)**2 for x in str(n)]) 
    slow, fast = n, n
    while True:
        slow, fast = next_num(slow), next_num(next_num(fast))
        if slow == 1: return True
        if slow == fast: return False 





H E L L O   :) 


Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. 
An island is surrounded by water and is formed by connecting adjacent 
lands horizontally or vertically. 
You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input:
11110
11010
11000
00000

Output: 1

Example 2:

Input:
11010
11110
00100
00011

Output: 2


Stuff ive already counted: 
(x, y)
0,0
0,1



Can only go in 4 directions 
What should I return if the input is [] 
Input will always be valid 


Going through the grid 
When I see a 1, I count it as an island, and keep going 
BUT, I dont wanna count any of the 1s that were touching it, ever again 

touching it, or touching one that touched it, and so on, forever 


starting at some X, Y cordinates, go UP , LEFT, DOWN, and RIGHT 
if its a 1, mark it/ keep track of it/ put it into our seen data structure 
and then keep going UP, LEFT, DOWN, RIGHT etc 
if its not a 1 (a 0 for instance), exit


*** 
[x0x00]
[xxxx0]
[00x00]
[00011]
*** 

0, 0 

def number_of_islands(grid): 
    island_count = 0
    for x_index, row in enumerate(grid):
        for y_index, square in enumerate(row):
            if square == 1: 
                island_count += 1
                mark_all_connected_stuff(x_index, y_index, grid)
    return island_count 

def mark_all_connected_stuff(x, y, grid): 
    places_to_go = [(x, y)]
    while places_to_go: 
        x, y = places_to_go.pop()
        # current_square is valid/in bounds/exsists and a 1
        if (x >= 0 and y >= 0 and 
           x < len(grid) and 
           y < len(grid[0]) and
           grid[x][y] == 1): 
            # marking this square as SEEN, explored, processed, previously visited, etc
            grid[x][y] = 'x'
            places_to_go.append((x + 1, y)) # DOWN
            places_to_go.append((x - 1, y)) # UP
            places_to_go.append((x, y + 1)) # RIGHT
            places_to_go.append((x, y - 1)) # LEFT


class GraphNode():
    __init__(self):
        self.value 
        self.children = []


def andrews_dfs_iterative(graph_node):
    stack = [graph_node]
    seen = set()
    while stack:
        current = stack.pop()
        if current not in seen: 
            seen.add(current)
            # DO SOME STUFF, whatever we want
            print(current.val)
            for child in current.children: 
                stack.append(child)

import deque from collections
def andrews_bfs_iterative(graph_node):
    q = deque()
    q.append(graph_node)
    seen = set()
    while q:
        current = q.popleft()
        if current not in seen: 
            seen.add(current)
            # DO SOME STUFF, whatever we want
            print(current.val)
            for child in current.children: 
                q.append(child)

def andrews_dfs_recursive(graph_node, seen = set()):
    if graph_node in seen: return
    seen.add(graph_node)
    print(graph_node.val)
    for child in graph_node.children:
        andrews_dfs_recursive(child, seen)

def moveZeroes(self, nums: List[int]) -> None:
    front = 0
    for idx, n in enumerate(nums): 
        if n != 0: 
            nums[front], nums[idx] = nums[idx], nums[front]
            front += 1


def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    anas = {}
    for word in strs:
        sw = ''.join(sorted(word))
        anas[sw] = anas.get(sw, []) + [word]
    return anas.values() 


       # 1.5 * n time, count the len first
def middleNode(self, head: ListNode) -> ListNode:
    cur, length = head, 0
    while cur: 
        length, cur = length + 1, cur.next
    for _ in range(length // 2): 
        head = head.next
    return head

# N time, Flyod's again
def middleNode(self, head: ListNode) -> ListNode:
    slow, fast = head, head
    while fast and fast.next: 
        slow, fast = slow.next, fast.next.next
    return slow


def backspaceCompare(self, S: str, T: str) -> bool:
    def get_letters(s):
        deletes = 0 
        for l in reversed(s): 
            if  l == '#':
                deletes += 1
            elif deletes > 0:
                deletes -= 1
            else:
                yield l
        yield ''
    S, T = get_letters(S), get_letters(T)
    ls, lt = next(S), next(T)
    while ls and lt: 
        if ls != lt:
            return False
        ls, lt = next(S), next(T)
    return not ls and not lt



# https://fellowship.hackbrightacademy.com/materials/challenges/sort-ab/index.html#sort-ab

# GIVEN:
# a = [1, 3, 4]
#         ^ 
# b = [2, 8, 10]
#      ^

# i_a = 1
# i_b = 0

# new_list = [1]

# we want to produce one new list out of these two lists, that is in order

# OUTPUT:
# we want to return [1, 2, 3, 5, 6, 7, 8, 10]

# new_list = a + b

# new_list = [1, 3, 5, 7, 2, 6, 8, 10]


# compare the front numbers in each list with each other
# take the smaller one out and add it to our new list 
# repeat until one of the lists is empty 
# add the other list to the back of our new list
# done

# # works, but its slow, O(n log n)
# def sort_two_lists(a, b):
#     return sorted(a + b)

# # works and is optimal, runtime is O(n)
# def sort_two_lists2(a, b):
#     new_list = []
#     i_a, i_b = 0, 0
#     while i_a < len(a) and i_b < len(b):
#         if a[i_a] <= b[i_b]:
#             new_list.append(a[a_i]) 
#             i_a += 1
#         else: 
#             new_list.append(b[i_b])
#             i_b += 1
#     return new_list + a[i_a:] + b[i_b:]


#     new_list = [1, 2, 3, 5, 6, 7] 
#     [1, 2, 3, 5, 6, 7] + [8, 10] + []



class MinStack:

    def __init__(self):
        self.s = []
        self.min = float("inf")
        
    def push(self, x: int) -> None:
        self.min = min(self.min, x)
        self.s.append((x, self.min))
        
    def pop(self) -> None:
        top = self.s.pop()[0]
        if self.s and self.s[-1][1] > self.min:
            self.min = self.s[-1][1]
        if not self.s: self.min = float("inf")
        return top 
    
    def top(self) -> int:
        return self.s[-1][0]

    def getMin(self) -> int:
        return self.min


def diameterOfBinaryTree(self, root: TreeNode) -> int:
    self.ans = 0 
    def max_depth(n):
        if not n: return 0
        right, left = max_depth(n.right), max_depth(n.left)
        self.ans = max(self.ans, left + right)
        return 1 + max(left, right)
    max_depth(root)
    return self.ans



import heapq
def lastStoneWeight(self, stones: List[int]) -> int:
    h = [n * -1 for n in stones]
    heapq.heapify(h)
    while len(h) > 1: 
        stone1, stone2 = heapq.heappop(h)*-1, heapq.heappop(h)*-1
        new_stone = max(stone1, stone2) - min(stone1, stone2)
        if new_stone: 
            heapq.heappush(h, new_stone * -1)
    return h[0] * -1 if h else 0

# n^2
def findMaxLength(self, nums: List[int]) -> int:
    best = 0
    for i, n in enumerate(nums):
        balance = 0
        for y in range(i, len(nums)):
            balance += 2 * nums[y] - 1
            if balance == 0:
                best = max(best, (y - i) + 1)
    return best

# n
def findMaxLength(self, nums: List[int]) -> int:
    current_balance, longest = 0, 0
    counts = {0: -1}
    for idx, n in enumerate(nums):
        # balance +1 for 1's and -1 for 0's
        # our balance is how many more 0's or 1's weve seen at this point
        current_balance += n and 1 or -1
        if current_balance in counts: 
            # if at any point we can look back and see the same balance in our dict
            # it means between our current location and that location 
            # we must have seen a balanced number of 0's and 1's
            # to have arrived back at the same current balance 
            # this this trick allows us to check every "sub array" weve seen before us in constant time
            longest = max(longest, idx - counts[current_balance])
        else:
            # if weve never had this number of 1's and 0's before then there is no valid subarray right now
            # so we just update our dict with the current balance and index
            counts[current_balance] = idx
    return longest

Input: s = "abcdefg", shift = [[1,1],[1,1],[0,2],[1,3]]
Output: "efgabcd"

[1 2 3 4 5]

[1,1] means shift to right by 1. "abcdefg" -> "gabcdef"
[1,1] means shift to right by 1. "gabcdef" -> "fgabcde"
[0,2] means shift to left by 2. "fgabcde" -> "abcdefg"
[1,3] means shift to right by 3. "abcdefg" -> "efgabcd"

# naive. do each shift as instructed, in order, by hand.
def stringShift(self, s: str, shift: List[List[int]]) -> str:
    q = deque(list(s)) 
    for s in shift: 
        #right
        if s[0]:
            for _ in range(s[1]+1):
                q.appendleft(q.pop())
        #left 
        else:
            for _ in range(s[1]+1):
                q.append(q.popleft())
    return ''.join(list(q))
  
# we need to notice 3 things to do better:

# 1) it doesnt matter what order we do the shifts in. this is the same as 
# order not mattering for + and - on a numberline
# because of this we can simply accumulate a single total for left and right moves

# 2) there is no point in making moves that end up canceling each other out, 
# so we only need to move left or right by the greater number of moves minus the lesser 

# 3) since we are rotating in circles, there is no point in any set of moves that results in a complete circle 
# as we would simply end up where we started. I.E. on a length 3 string moving 1 has the same outcome as moving 4
# so we can mod the amount we are to move by the length of the string to avoid unessecary moves

def stringShift(self, s: str, shift: List[List[int]]) -> str:
  total_moves = {0:0, 1:0}
  q = deque(list(s)) 
  for move, amt in shift: 
      total_moves[move] =  total_moves[move] + amt 
  if total_moves[0] > total_moves[1]:
      amt = total_moves[0] - total_moves[1]
      amt = amt % len(s)
      for _ in range(amt):
          q.append(q.popleft())
  if total_moves[1] > total_moves[0]:
      amt = total_moves[1] - total_moves[0]
      amt = amt % len(s)
      for _ in range(amt):
          q.appendleft(q.pop())
  return ''.join(list(q))

def stringShift(self, s: str, shift: List[List[int]]) -> str:
  total_moves = {0:0, 1:0}
  for move, amt in shift:
      total_moves[move] += amt 
# left
  if total_moves[0] > total_moves[1]:
      amt = total_moves[0] - total_moves[1]
      amt = amt % len(s)
      s = s[:-amt] + s[-amt:]
# right
  if total_moves[1] > total_moves[0]:
      amt = total_moves[1] - total_moves[0]
      amt = amt % len(s)
      s = s[-amt:] + s[:-amt]
  return s

def stringShift(self, s: str, shift: List[List[int]]) -> str:
    total_moves = {0:0, 1:0}
    for move, amt in shift:
        total_moves[move] += amt 
    amt = abs(total_moves[0] - total_moves[1])
    amt = amt % len(s)
    return s[-amt:] + s[:-amt]

# since were just going to make a single decisive move now, we dont need the deque or loops 
def stringShift(self, s: str, shift: List[List[int]]) -> str:
    total_move = 0
    for move, amt in shift:
        if move:
            total_move += amt
        else:
            total_move -= amt
    amt %= len(s)
    return s[-amt:] + s[:-amt]

def stringShift(self, s: str, shift: List[List[int]]) -> str:
    i = sum([a, -a][d] for d, a in shift) % len(s)
    return s[s:] + s[:s]

def productExceptSelf(self, nums: List[int]) -> List[int]:
    if len(nums) == 1: return nums      
    # the major trick youre supposed to take away from this problem is called prefix sum (or product)
    # we can preprocess some of our information, in this case products, both forwards and backwards
    # into two different arrays like so: 
    # original = 1 2 3 4
    # forward product array = 1 2 6 24 
    # backward product array = 24 24 12 4
    # for our answer, since it is the product of every number except the one youre on
    # we can look to the product to our left in the forward array
    # and the product to our right in the backward array 
    # and multiply them together for the answer in any given position:
    # answer[current] = forward[current - 1] * backward[current + 1]
    
    # there is a special case in that the end numbers should just be * 1, so here is my solution
    # create a prefix product array, but starting with 1 and ending one product early
    # then run a backwards product on our nums in the same way,
    # starting with a 1 and ending a product early
    # our sequences look like this:
    # 1  2  3 4  == original 
    # 1  1  2 6  --> forwards prod starting with a 1 and ending early
    # 24 12 4 1  <-- backwards prod starting with a 1 and ending early
    # 24 12 8 6  == answer 
    # we can update our array as we go by multiplying the two together 
    
    scan, prod = [], 1
    for n in nums:
        scan.append(prod)
        prod *= n
        
    prod = 1
    for i in range(len(nums)-1, -1, -1):
        scan[i] *= prod
        prod *= nums[i]
    return scan


# n^3 recursive with memo, passes 55/58 test cases, times out
def checkValidString(self, s: str) -> bool:
    def check(idx, count):
        key = (idx, count)
        if key in memo:
            return memo[key]
        for i in range(idx, len(s)): 
            c = s[i]
            if c == '(':
                count += 1
            elif c == '*':
                a = check(i+1, count-1)
                b = check(i+1, count+1)
                c = check(i+1, count)
                return a or b or c
            else: 
                count -= 1
                if count < 0: 
                    memo[key] = False
                    return False
        memo[key] = count == 0
        return count == 0
    
    memo = {}
    return check(0, 0)
                
# hours of hating yourself later, you get this:
def checkValidString(self, s: str) -> bool:
    L = R = 0
    for i in range(len(s)):
        L += 1 if s[i] in "(*" else -1 
        R += 1 if s[len(s)-i-1] in "*)" else -1
        if L < 0  or R < 0: return False
    return True



# we could try every path and take the min but the runtime sucks 
# being greedy is vulerable to edge cases 
# we could use a priority Q and do better, but still not optimal 
# luckily, the fact we can only move down or right makes precomputing the optimal cost of arrival 
# at each location pretty easily 
# compute the prefix sum of the top row going right 
# compute the prefix sum of the first column going down 
# finally, at each square take the min of having come from UP or from LEFT
# this is DP and can be best visualized by drawing out the grid we construct 
# once were done, whatever the bottom right value is is guranteed to be the lowest sum path
# O(n) time, constant space

def minPathSum(self, grid: List[List[int]]) -> int:
    for i in range(1, len(grid[0])):
        grid[0][i] += grid[0][i-1]
    for i in range(1, len(grid)):
        grid[i][0] += grid[i-1][0]
    for x in range(1, len(grid)):
        for y in range(1, len(grid[0])):
            grid[x][y] += min(grid[x-1][y], grid[x][y-1])
    return grid[-1][-1]

def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
    if not preorder: return
    root, i = TreeNode(preorder.pop(0)), 0
    while i < len(preorder) and preorder[i] < root.val: 
        i += 1
    root.left = self.bstFromPreorder(preorder[:i])
    root.right = self.bstFromPreorder(preorder[i:])
    return root

def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
    # binary search each row
    # our target is a space that is a 1 AND to the right of a 0
    # making it the left most 1 
    def firstOne(row):
        l, r = 0, width - 1
        while r >= l:
            mid = l + (r - l) // 2
            n = binaryMatrix.get(row, mid)   
            if n == 1 and mid == 0 or n == 1 and binaryMatrix.get(row, mid-1) == 0:
                return mid
            if n == 1: 
                r = mid - 1
            else: 
                l = mid + 1 
        return 101
    
    height, width = binaryMatrix.dimensions()
    leftest = 101
    for row in range(height):
        leftest = min(firstOne(row), leftest)
    return leftest if leftest < 101 else -1

def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
height, width = binaryMatrix.dimensions()
row, col, last = 0, width - 1, -1
while row < height and col > -1:
    p = binaryMatrix.get(row, col)   
    if p == 0: 
        row += 1
    else: 
        last = col
        col -= 1
return last


def maximalSquare(self, matrix: List[List[str]]) -> int:
    def is_square(top_x, top_y, b_x, b_y):
        for i in range((b_x - top_x) +1):
            if sum(matrix[top_x + i][top_y:b_y+1]) != (b_x - top_x) + 1:
                return False
        return True
    
    def max_square(x, y):
        org_x, org_y = x, y
        best = 1
        while x < len(matrix) and y < len(matrix[0]):
            if not is_square(org_x, org_y, x, y): 
                break
            best = (x + 1 - org_x)**2
            x, y = x+1, y+1
        return best

    for i in range(len(matrix)):
        matrix[i] = [int(n) for n in matrix[i]]
    
    max_so_far = 0
    for ri, row in enumerate(matrix): 
        for si, square in enumerate(row): 
            if square == 1: 
                max_so_far = max(max_so_far, max_square(ri, si))
    return max_so_far 


class FirstUnique:

    def __init__(self, nums: List[int]):
        self.q = nums
        self.c = {}
        for n in nums:
            self.c[n] = self.c.get(n, 0) + 1
        
    def showFirstUnique(self) -> int:
        while self.q and self.q[0] in self.c and self.c[self.q[0]] > 1:
            self.q.pop(0)
        return self.q[0] if self.q else -1 
    
    def add(self, value: int) -> None:
        self.c[value] = self.c.get(value, 0) + 1
        self.q.append(value)


def maxPathSum(self, root: TreeNode) -> int:
    def helper(root):
        if not root: 
            return [float('-inf'), float('-inf')] 
        left_best, left_terminal_best = helper(root.left)
        right_best, right_terminal_best = helper(root.right)
        current_only = root.val 
        current_and_left = root.val + left_best
        current_and_right = root.val + right_best
        current_and_both_sides = root.val + right_best + left_best
        best = max(current_only, current_and_left, current_and_right)
        terminal_best = max(left_terminal_best, left_best, right_best, right_terminal_best, current_and_both_sides)
        return [best, terminal_best]
    return max(helper(root))
    
    
# but thats way too easy to read right? i got you 
def maxPathSum(self, root: TreeNode) -> int:
    def helper(root):
        if not root: return [float('-inf'), float('-inf')] 
        lb, lt = helper(root.left)
        rb, rt = helper(root.right)
        return [max(root.val, root.val+lb , root.val+rb), max(lt, lb, rb, rt, root.val+rb+lb)]
    return max(helper(root))


def isValidSequence(self, root: TreeNode, arr: List[int]) -> bool:
    if not root or not arr or root.val != arr[0]: 
        return False 
    if len(arr) == 1 and not root.left and not root.right: 
        return True
    l = self.isValidSequence(root.left, arr[1:])            
    r = self.isValidSequence(root.right, arr[1:])   
    return l or r

def firstValidVersion(n):
    if n == 1: return n
    lo, hi = 1, n
    while lo <= hi: 
        cur = lo + ((hi - lo) // 2)
        bad = isBadVersion(cur)
        if bad and (cur == 0 or not isBadVersion(cur-1)):
            return cur
        if bad: 
            hi = cur - 1
        else:   
            lo = cur + 1

import collections
def firstUniqChar(self, s: str) -> int:
    seen = collections.Counter(s)
    for i, l in enumerate(s): 
        if seen[l] == 1:
            return i
    return -1



class Dog: 
    hunger_level = 10
    swims = True
    diet = "pescitarian"
    name = None
    species = 'dog'

    def speak(self):
        print(f'Woof! I am {self.name} the {self.species}')

    def graduate(self):
        self.name = f'Dr. {self.name}'

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.t = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        if not word: return
        cur = self.t
        for l in word: 
            if not cur.get(l):
                cur[l] = {}
            cur = cur[l]
        cur['isWord'] = True
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if not word: return
        cur = self.t
        for l in word:
            if l not in cur: 
                return False 
            cur = cur[l]
        return 'isWord' in cur

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        if not prefix: return
        cur = self.t
        for l in prefix:
            if l not in cur: 
                return False 
            cur = cur[l]
        return True


# https://leetcode.com/problems/surrounded-regions/
def solve(self, board: List[List[str]]) -> None:
    def follow(x, y):
        if x < 0 or y < 0: return 
        if x >= len(board) or y >= len(board[0]): return 
        if board[x][y] != 'O': return 
        board[x][y] = '*'
        follow(x+1, y)
        follow(x, y+1)
        follow(x-1, y)
        follow(x, y-1)
        
    for x, row in enumerate(board):
        for y, cell in enumerate(row): 
            if (cell == 'O' and 
                (x == 0 or y == 0 or 
                x == len(board)-1 or y == len(row)-1)):
                    follow(x, y)
    
    for x, row in enumerate(board):
        for y, cell in enumerate(row): 
            if cell == 'O':
                board[x][y] = 'X'
            if cell == '*':
                board[x][y] = 'O'
    
    return board


import pickle 
example_dict = {1:"6",2:"2",3:"f"}
mystery = pickle.dumps(example_dict)
print(mystery)
dict2 = pickle.loads(mystery)
print(dict2)


import random
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.l = []
        self.h = {}
        

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.h: return False
        self.l.append(val)
        self.h[val] = len(self.l) - 1
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.h: return False
        idx = self.h[val]
        del self.h[val]
        if idx != len(self.l) - 1:
            self.l[idx], self.l[len(self.l) -1] = self.l[len(self.l) -1], self.l[idx]
            self.h[self.l[idx]] = idx
        self.l.pop()
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return random.choice(self.l)

def runningSum(self, nums: List[int]) -> List[int]:
    for i in range(1, len(nums)):
        nums[i] = nums[i] + nums[i-1]
    return nums

def arrangeCoins(self, n: int) -> int:
    counter = 1
    total = 0
    while n >= counter: 
        total += 1
        n -= counter
        counter += 1
    return total    


def prisonAfterNDays(self, cells: List[int], N: int) -> List[int]:
    for i in range((N - 1) % 14 + 1):
#             copy = cells[:]
#             for c in range(1, len(cells)-1):
#                 copy[c] = int(cells[c-1] == cells[c+1])
#             cells = [0] + copy[1:-1] + [0]
        cells = [0] + [int(cells[i-1] == cells[i+1]) for i in range(1,7)] + [0]
    return cells

def plusOne(self, digits: List[int]) -> List[int]:
    for i in range(len(digits)-1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits 
        else: 
            digits[i] = 0
    return [1] + digits

def widthOfBinaryTree(self, root: TreeNode) -> int:
    m = 0 
    q = [(root, 0)]
    while q: 
        m = max(m, q[-1][1] - q[0][1] + 1)
        n = []
        for child, number in q: 
            if child.right: n.append((child.right, 2 * number))
            if child.left: n.append((child.left, (2 * number) + 1))
        q = n
    return m

def subsets(self, nums: List[int]) -> List[List[int]]:
    result = [[]]
    for n in nums: 
        result += [[n] + i for i in result]
    return result


class FlattenSolution:
    def flatten(self, head: 'Node') -> 'Node':
        h, t = self.flatten2(head)
        return h
    
    def flatten2(self, head: 'Node') -> 'Node':
        cur = head
        prev = head
        while cur:
            if cur.child: 
                stored_next = cur.next
                cur.next = cur.child 
                cur.child.prev = cur 
                _, end = self.flatten2(cur.child)
                cur.child = None
                end.next = stored_next 
                if stored_next:
                    stored_next.prev = end
            prev = cur
            cur = cur.next
        return (head, prev)

def angleClock(self, hour: int, minutes: int) -> float:
    if hour == 12:
        hour = 0 
    hour = (hour * 30) + (minutes * 0.5)
    minutes = minutes * 6
    dif = abs(hour - minutes)
    return min(dif, 360 - dif)
        


# highlight, click, double click, paren matching 
# definition, peek
# autocomplete 

# CMD F 
# CMD SHIFT F 
# CMD SHIFT T 

# open and close side panel with CMD B 
# open and close side panel with CMD J

# OPTION up and down
# OPTION SHIFT up and down 
