# Comprehensive Competitive Programming Cheatsheet

# Fast input/output template for Competitive Programming

import sys
from typing import List, Set, Dict, Tuple
from collections import defaultdict, Counter, deque
from heapq import heappush, heappop, heapify
import math
import bisect
import itertools

# For faster input
input = sys.stdin.readline
# For faster printing
print = sys.stdout.write

def solve():
   # Single integer
   n = int(input())
   
   # Multiple integers on one line
   a, b, c = map(int, input().split())
   
   # List of integers
   arr = list(map(int, input().split()))
   
   # Multiple lines of input
   grid = []
   for _ in range(n):
       row = list(map(int, input().split()))
       grid.append(row)
   
   # String input
   s = input().strip()  # strip() removes trailing newline
   
   # Output formats
   print(f"{answer}\n")  # Don't forget \n
   print(" ".join(map(str, arr)) + "\n")  # Print array
   print(f"{floating_point:.10f}\n")  # Print float with precision

def main():
   # For single test case
   solve()
   
   # For multiple test cases
   t = int(input())
   for _ in range(t):
       solve()

if __name__ == "__main__":
   main()

"""
Template Usage:

Problem with single test case:
5
1 2 3 4 5

Problem with multiple test cases:
2
3
1 2 3
4
1 2 3 4

Grid input:
3 3
1 2 3
4 5 6
7 8 9
"""


## Section 1: Algorithm Selection Guide

### How to Choose the Right Algorithm

When you first see a problem, look for these key indicators:

1. Input Size (n) - This is your first clue:
   ```
   n ≤ 10         → O(n!) or O(n^6) is OK
   n ≤ 20         → O(2^n) is OK
   n ≤ 500        → O(n^3) is OK
   n ≤ 5000       → O(n^2) is OK
   n ≤ 100,000    → O(n log n) is OK
   n ≤ 1,000,000  → O(n) is OK
   n > 1,000,000  → O(log n) or O(1) needed
   ```

2. Problem Type Indicators:
   - "Find minimum/maximum" → Think Dynamic Programming or Greedy
   - "Find shortest path" → Think BFS (unweighted) or Dijkstra (weighted)
   - "Find in sorted array" → Think Binary Search
   - "Subarray/substring" → Think Sliding Window
   - "Tree/Graph traversal" → Think DFS/BFS
   - "Connected components" → Think Union-Find or DFS
   - "All possible combinations" → Think Backtracking

3. Data Structure Selection:
   - Need fast lookup? → Hash Map (dict)
   - Need ordered elements? → Heap or BST
   - Need to track frequencies? → Counter
   - Need quick insertions/deletions at ends? → Deque
   - Need to find groups? → Union-Find
   - Need to track intervals? → Segment Tree

## Section 2: Essential Python Libraries

### Collections Module
```python
from collections import defaultdict, Counter, deque

# defaultdict - Dictionary with default value
graph = defaultdict(list)    # Default: empty list
freq = defaultdict(int)      # Default: 0
graph[1].append(2)          # No KeyError if key doesn't exist

# Counter - Count frequencies
nums = [1, 1, 2, 3, 3, 3]
count = Counter(nums)
print(count)                # Counter({3: 3, 1: 2, 2: 1})
print(count.most_common(2)) # [(3, 3), (1, 2)]

# deque - Double-ended queue
dq = deque([1, 2, 3])
dq.append(4)        # Add to right: [1,2,3,4]
dq.appendleft(0)    # Add to left: [0,1,2,3,4]
dq.pop()            # Remove from right: [0,1,2,3]
dq.popleft()        # Remove from left: [1,2,3]
dq.rotate(1)        # Rotate right: [3,1,2]
```

### Heapq Module
```python
from heapq import heappush, heappop, heapify

# Min heap operations
heap = []
heappush(heap, 5)   # Add element
smallest = heappop(heap)  # Remove smallest

# Convert list to heap
nums = [3, 1, 4, 1, 5]
heapify(nums)       # Transforms list into heap in-place

# For max heap, negate values
max_heap = []
heappush(max_heap, -5)  # Add -5
largest = -heappop(max_heap)  # Get 5

# Priority queue with tuples
pq = []
heappush(pq, (priority, item))
```

### Bisect Module
```python
from bisect import bisect_left, bisect_right

arr = [1, 2, 2, 2, 3, 4]
# Find insertion point
pos = bisect_left(arr, 2)   # pos = 1
pos = bisect_right(arr, 2)  # pos = 4

# Binary search
def binary_search(arr, x):
    i = bisect_left(arr, x)
    return i if i != len(arr) and arr[i] == x else -1
```

## Section 3: Core Algorithms

### Binary Search
```python
def binary_search(arr, target):
    """
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Binary search on answer space
def binary_search_predicate(left, right, condition):
    while left < right:
        mid = (left + right) // 2
        if condition(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

Use Binary Search when:
- Searching in sorted array
- Finding minimum value satisfying condition
- Optimizing max/min problems
- Finding closest element

Common Mistakes:
- Using `<` instead of `<=` in while loop
- Integer overflow in mid calculation
- Not handling duplicates properly
- Wrong boundary updates

### Depth-First Search (DFS)
```python
def dfs_recursive(graph, node, visited=None):
    """
    Time: O(V + E)  V = vertices, E = edges
    Space: O(V)
    """
    if visited is None:
        visited = set()
        
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    return visited

def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(neighbor for neighbor in graph[node]
                        if neighbor not in visited)
    return visited
```

Use DFS when:
- Finding paths in graph
- Detecting cycles
- Topological sorting
- Exploring all possibilities
- Tree traversal

Common Mistakes:
- Forgetting to mark nodes as visited
- Stack overflow in recursive version
- Not handling disconnected components
- Infinite loops in cyclic graphs

### Breadth-First Search (BFS)
```python
from collections import deque

def bfs(graph, start):
    """
    Time: O(V + E)
    Space: O(V)
    """
    visited = set([start])
    queue = deque([start])
    level = {start: 0}
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                level[neighbor] = level[node] + 1
    return level
```

Use BFS when:
- Finding shortest path (unweighted)
- Level-order traversal
- Finding closest nodes
- Testing bipartite graphs

Common Mistakes:
- Using stack instead of queue
- Not tracking levels/distances properly
- Memory overflow in large graphs
- Not checking visited nodes

### Sliding Window
```python
def fixed_window(arr, k):
    """
    Time: O(n)
    Space: O(1)
    """
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum

def variable_window(s):
    """
    Time: O(n)
    Space: O(k) - k = distinct elements
    """
    left = max_len = 0
    seen = {}
    
    for right in range(len(s)):
        if s[right] in seen:
            left = max(left, seen[s[right]] + 1)
        seen[s[right]] = right
        max_len = max(max_len, right - left + 1)
    return max_len
```

Use Sliding Window when:
- Processing subarrays/substrings
- Finding longest/shortest sequence
- Maintaining running statistics
- Pattern matching

Common Mistakes:
- Off-by-one errors in window bounds
- Not updating window properly
- Wrong window size calculation
- Incorrect window condition checks

### Dynamic Programming
```python
# 1D Dynamic Programming
def fibonacci_dp(n):
    """
    Time: O(n)
    Space: O(n)
    """
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 2D Dynamic Programming
def longest_common_subsequence(text1, text2):
    """
    Time: O(mn)
    Space: O(mn)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

Use Dynamic Programming when:
- Problem has overlapping subproblems
- Need optimal solution
- Can be broken into smaller problems
- Has recursive relation

Common Mistakes:
- Wrong base cases
- Incorrect state transition
- Not handling edge cases
- Memory limit exceeded

### Union-Find (Disjoint Set)
```python
class UnionFind:
    """
    Time: O(α(n)) ≈ O(1) per operation
    Space: O(n)
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

Use Union-Find when:
- Finding connected components
- Detecting cycles in undirected graph
- Minimum spanning tree
- Dynamic connectivity

Common Mistakes:
- Not using path compression
- Not using union by rank
- Wrong parent updates
- Not checking for existing connections

## Section 4: Graph Algorithms

### Dijkstra's Algorithm
```python
from heapq import heappush, heappop

def dijkstra(graph, start):
    """
    Time: O((V + E) log V)
    Space: O(V)
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, node = heappop(pq)
        
        if d > distances[node]:
            continue
            
        for neighbor, weight in graph[node].items():
            distance = d + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(pq, (distance, neighbor))
    
    return distances
```

Use Dijkstra when:
- Finding shortest path in weighted graph
- Path with minimum cost
- Network routing
- GPS navigation

Common Mistakes:
- Using negative weights
- Not checking for better paths
- Wrong priority queue usage
- Memory overflow in dense graphs

## Section 5: Tree Algorithms

### Binary Tree Traversal
```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

def inorder(root):
    """
    Time: O(n)
    Space: O(h) - h = height
    """
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# Iterative inorder
def inorder_iterative(root):
    result = []
    stack = []
    curr = root
    
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    
    return result
```

Use Tree Traversal when:
- Processing tree nodes in specific order
- Tree validation
- Tree transformation
- Finding paths

Common Mistakes:
- Not handling empty tree
- Wrong recursion order
- Stack overflow
- Not maintaining proper state

## Section 6: String Algorithms

### String Matching (KMP)
```python
def build_lps(pattern):
    """
    Build Longest Proper Prefix which is also Suffix array
    Time: O(m) - m = pattern length
    """
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    """
    Time: O(n + m)
    Space: O(m)
    """
    if not pattern:
        return []
    
    lps = build_lps(pattern)
    results = []
    
    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            results.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return results
```

Use KMP when:
- Pattern matching in strings
- Finding all occurrences
- Text processing
- DNA sequence matching

Common Mistakes:
- Wrong LPS array construction
- Index management errors
- Not handling edge cases
- Inefficient pattern comparison

## Section 7: Common Problem-Solving Patterns

### Two Pointers (continued)
```python
def two_sum_sorted(arr, target):
    """
    Time: O(n)
    Space: O(1)
    
    Use for: Finding pairs in sorted array that sum to target
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    return [-1, -1]

def remove_duplicates(arr):
    """
    Time: O(n)
    Space: O(1)
    
    Use for: Removing duplicates in-place from sorted array
    """
    if not arr:
        return 0
        
    write_pos = 1  # Position to write next unique element
    
    for read_pos in range(1, len(arr)):
        if arr[read_pos] != arr[read_pos-1]:
            arr[write_pos] = arr[read_pos]
            write_pos += 1
            
    return write_pos
```

When to Use Two Pointers:
1. Sorted array problems
2. Finding pairs with sum/difference conditions
3. In-place array modifications
4. Palindrome problems
5. Merging sorted arrays
6. Finding subarrays with conditions

Common Mistakes:
- Not checking boundary conditions
- Wrong pointer movement logic
- Not handling duplicates properly
- Infinite loops due to wrong termination condition

## Section 8: Advanced Data Structures

### Trie (Prefix Tree)
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    """
    Time Complexity:
    - Insert: O(m) where m = word length
    - Search: O(m)
    - Prefix Search: O(m)
    
    Space: O(ALPHABET_SIZE * m * n) where n = number of words
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

Use Trie when:
- Word dictionary implementation
- Prefix matching
- Auto-complete features
- Spell checker
- IP routing tables

### Segment Tree
```python
class SegmentTree:
    """
    Time Complexity:
    - Build: O(n)
    - Query: O(log n)
    - Update: O(log n)
    
    Space: O(n)
    """
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)  # 4n size is enough
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
            return
            
        mid = (start + end) // 2
        self.build(arr, 2*node + 1, start, mid)
        self.build(arr, 2*node + 2, mid + 1, end)
        self.tree[node] = min(self.tree[2*node + 1], 
                             self.tree[2*node + 2])
    
    def query(self, node, start, end, l, r):
        if start > r or end < l:
            return float('inf')
        if l <= start and end <= r:
            return self.tree[node]
            
        mid = (start + end) // 2
        left = self.query(2*node + 1, start, mid, l, r)
        right = self.query(2*node + 2, mid + 1, end, l, r)
        return min(left, right)
    
    def update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
            
        mid = (start + end) // 2
        if idx <= mid:
            self.update(2*node + 1, start, mid, idx, val)
        else:
            self.update(2*node + 2, mid + 1, end, idx, val)
        self.tree[node] = min(self.tree[2*node + 1], 
                             self.tree[2*node + 2])
```

Use Segment Tree when:
- Range queries (min, max, sum)
- Range updates
- Dynamic range problems
- Finding frequent elements in ranges

## Section 9: Math and Number Theory

### Prime Numbers
```python
def sieve_of_eratosthenes(n):
    """
    Generate all primes up to n
    Time: O(n log log n)
    Space: O(n)
    """
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            for j in range(i*i, n+1, i):
                primes[j] = False
                
    return [i for i in range(n+1) if primes[i]]

def prime_factors(n):
    """
    Find all prime factors of n
    Time: O(sqrt(n))
    """
    factors = []
    i = 2
    
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 1
    
    if n > 1:
        factors.append(n)
    return factors
```

### GCD and LCM
```python
def gcd(a, b):
    """Greatest Common Divisor using Euclidean algorithm"""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Least Common Multiple"""
    return abs(a * b) // gcd(a, b)

def extended_gcd(a, b):
    """
    Returns (gcd, x, y) where ax + by = gcd
    Used for finding modular multiplicative inverse
    """
    if a == 0:
        return b, 0, 1
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b//a) * x1
    y = x1
    
    return gcd, x, y
```

## Section 10: Common Tricks and Optimizations

### Fast Input/Output
```python
import sys
input = sys.stdin.readline
print = sys.stdout.write

# Fast integer input
n = int(input())
# Fast array input
arr = list(map(int, input().split()))
```

### Bit Manipulation
```python
# Check if number is power of 2
def is_power_of_two(n):
    return n > 0 and (n & (n-1)) == 0

# Count set bits
def count_set_bits(n):
    count = 0
    while n:
        n &= (n-1)  # Clear least significant bit
        count += 1
    return count

# Get/Set/Clear bit
def get_bit(num, i):
    return (num >> i) & 1
def set_bit(num, i):
    return num | (1 << i)
def clear_bit(num, i):
    return num & ~(1 << i)
```

### Python Built-in Functions
```python
# Number related
bin(n)      # Convert to binary string
oct(n)      # Convert to octal string
hex(n)      # Convert to hexadecimal string
abs(n)      # Absolute value
pow(x,y,p)  # Compute (x^y) % p efficiently

# Sequence operations
reversed(seq)   # Reverse iterator
sorted(seq)     # Return sorted list
any(iterable)   # True if any element is True
all(iterable)   # True if all elements are True
zip(it1, it2)   # Pair corresponding elements

# String operations
ord('a')        # Get ASCII value
chr(97)         # Get character from ASCII
str.join(seq)   # Join sequence with str
```

## Section 11: Contest Strategy Tips

1. Problem Analysis
   - Read all problems first
   - Sort by difficulty
   - Look for patterns in sample cases
   - Check constraints carefully

2. Time Management
   - Don't get stuck on one problem
   - Skip if stuck for >30 minutes
   - Save complex problems for last
   - Leave time for debugging

3. Testing Strategy
   - Test edge cases first
   - Small test cases manually
   - Generate large test cases
   - Test corner cases:
     - Empty input
     - Single element
     - All same elements
     - Maximum/minimum values

4. Common Mistakes to Avoid
   - Integer overflow
   - Array bounds
   - Uninitialized variables
   - Off-by-one errors
   - Stack overflow
   - Infinite loops
   - Wrong order of operations

5. Debugging Tips
   - Print intermediate values
   - Use assert statements
   - Binary search the bug
   - Check algorithm assumptions
   - Verify data structure states

## Section 12: Problem Categories and Patterns

1. Array Problems
   - Two pointers
   - Sliding window
   - Prefix sum
   - Binary search
   - Sorting

2. String Problems
   - Two pointers
   - Sliding window
   - Trie
   - KMP
   - Dynamic programming

3. Graph Problems
   - DFS/BFS
   - Shortest path
   - Cycle detection
   - Topological sort
   - Connected components

4. Tree Problems
   - DFS/BFS
   - Binary search tree
   - Path finding
   - Level order traversal
   - Lowest common ancestor

5. Dynamic Programming
   - 0/1 Knapsack
   - Longest common subsequence
   - Edit distance
   - Matrix chain multiplication
   - Coin change

Remember:
- Start with simple solution
- Optimize only if needed
- Test thoroughly
- Learn from mistakes
- Practice regularly



# Understanding Algorithms Through Everyday Metaphors

## Searching Algorithms

### Binary Search
Imagine you're searching for a book in a massive library where all books are sorted by title. Instead of checking every single book, you start in the middle of the library. If your book comes earlier in the alphabet, you move to the middle of the first half; if it comes later, you move to the middle of the second half. It's like playing a number guessing game where someone tells you "higher" or "lower" after each guess. Each time you look, you eliminate half of the remaining books, making it incredibly efficient. This is why binary search is so powerful – with just 20 steps, you could find one book among a million!

### Linear Search
Think of linear search as searching for your keys by checking every pocket in every piece of clothing you own, one by one. While it's not the most efficient method, it's straightforward and sometimes necessary when things aren't organized in any particular way. It's like being a detective who must examine every piece of evidence without any shortcuts.

## Graph Traversal Algorithms

### Breadth-First Search (BFS)
Imagine you're dropping a pebble in a still pond. The ripples spread out in perfect circles, reaching points that are closest to the center first, then gradually expanding outward. This is exactly how BFS works – it explores a graph level by level, visiting all nearby nodes before moving to more distant ones. It's particularly useful when you need to find the shortest path, just like how water will always find the quickest route to flow.

### Depth-First Search (DFS)
Think of DFS as exploring a maze with a single piece of string. You keep going as deep as possible down each path, leaving your string behind to mark your trail. When you hit a dead end, you backtrack along your string until you find an unexplored passage. This methodical exploration ensures you won't miss any passages, just as DFS won't miss any nodes in a graph.

## Tree-Based Algorithms

### Tree Traversal
Trees in computer science are like family trees, but turned upside down. Different ways of traversing them are like different ways of reading a family story. Inorder traversal is like reading a book from left to right. Preorder traversal is like a parent introducing their family – they state their name first, then introduce their children. Postorder traversal is like cleaning a house – you clean the children's rooms before the parent's room.

## Dynamic Programming

Dynamic programming is like planning a road trip with multiple possible routes. Instead of recalculating the time and distance for common route segments repeatedly, you write down the information for each segment once and reuse it. It's similar to a master chef who preps ingredients once and uses them in multiple dishes, rather than starting from scratch each time. The "dynamic" part comes from building up solutions to bigger problems using solutions to smaller problems you've already solved.

## Sorting Algorithms

### Merge Sort
Imagine you're organizing a giant stack of papers. Merge sort is like having many helpers who each take a small stack, sort it, and then help combine the sorted stacks. It's similar to how you might sort a deck of cards by splitting it into smaller piles, sorting each pile, and then merging them back together. This divide-and-conquer approach makes handling large amounts of data much more manageable.

### Quick Sort
Quick sort is like organizing a classroom of students by height. You pick one student (the pivot) and ask everyone else to stand either to their left (if shorter) or right (if taller). Then you repeat this process with each group until everyone is in order. It's fast because multiple comparisons happen simultaneously, just like how students can quickly figure out which side they should be on.

## Optimization Algorithms

### Greedy Algorithms
A greedy algorithm is like a hungry person at a buffet who always takes the most appealing dish they can see right now, without planning ahead. While this might not always lead to the best overall meal combination, it often works well enough and is very quick. In computer science, it's useful when we need quick decisions and a "good enough" solution is acceptable.

### Dijkstra's Algorithm
Think of Dijkstra's algorithm as a very cautious traveler planning a road trip. At each intersection, they consider all possible routes but always choose to explore the shortest total distance first. They keep track of all the distances they've found so far, constantly updating their map with better routes when they find them. It's like having a GPS that's determined to find not just any route, but the absolute shortest one.

## Data Structures

### Hash Tables
A hash table is like a library's card catalog system. Instead of searching through every book, you can look up exactly where a book is stored using a special code (the hash). It's similar to how a post office uses zip codes to quickly sort and deliver mail – the zip code tells you exactly which area to look in, making the process much faster.

### Heaps
A heap is like a company's organizational chart, but with a special rule: each manager must be more senior (or junior, depending on the type of heap) than their direct reports. It's particularly useful for always knowing who the most or least senior person is, just like how a priority queue always knows what task is most important.

## Pattern Matching Algorithms

### Sliding Window
The sliding window technique is like looking through a moving frame at a painting. Instead of studying the entire painting at once, you look through a frame that shows you a fixed portion, then slide that frame across the painting. This helps you focus on specific parts while maintaining context, just like how a photographer might frame different portions of a landscape.

### KMP (Knuth-Morris-Pratt)
KMP is like a smart reader who remembers patterns in text. Instead of starting over from the beginning when a mismatch occurs, they remember what they've seen and jump to the next possible match position. It's similar to how you might search for a word in a document – if you see "programm" but the next letter isn't "i", you don't need to start over completely because you might already be part way through another occurrence of "programming".

By understanding these metaphors, you can better grasp when to use each algorithm. Just as you wouldn't use a sledgehammer to hang a picture frame, choosing the right algorithm for the right situation is crucial for efficient problem-solving.
