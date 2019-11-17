
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