
// INPUT:
let a = [
  { startTime: 0,  endTime: 1 },
  { startTime: 3,  endTime: 5 },
  { startTime: 4,  endTime: 8 },
  { startTime: 10, endTime: 12 },
  { startTime: 9,  endTime: 10 },
]

let b = [
  { startTime: 1, endTime: 10 },
  { startTime: 2, endTime: 6 },
  { startTime: 3, endTime: 5 },
  { startTime: 7, endTime: 9 },
]

// OUTPUT:
// answer_a = [
//   { startTime: 0, endTime: 1 },
//   { startTime: 3, endTime: 8 },
//   { startTime: 9, endTime: 12 },
// ]

// answer_b = [{ startTimeL: 0, endTime: 12}]

function mergeRanges(meetings) {
  meetings.sort((a, b) => a.startTime - b.startTime)
  let overlaps = [meetings[0]]
  for (let i = 1; i < meetings.length; i++) {
    let currentMeeting = meetings[i]
    let lastMeeting = overlaps[overlaps.length - 1]
    if (currentMeeting.startTime > lastMeeting.endTime) {
      overlaps.push(currentMeeting)
    } else {
      lastMeeting.endTime = Math.max(currentMeeting.endTime, lastMeeting.endTime)
    }
  }
  return overlaps
}

// console.log(mergeRanges(a))
// console.log(mergeRanges(b))


// Reverse an array in place 

function rString(str) {
  let start = 0;
  let end = str.length - 1 
  while (start < end) {
    let temp = str[start]
    str[start] = str[end]
    str[end] = temp
    start += 1
    end -= 1
  }
}

// console.log(rString(['a','n','d']))
// console.log(rString(['a','b']))

const message = [ 'c', 'a', 'k', 'e', ' ',
'p', 'o', 'u', 'n', 'd', ' ',
's', 't', 'e', 'a', 'l' ];


function reverseWords(words) {
  reverseSingleWord(words, 0, words.length - 1)
  firstLetter = 0;
  for (let i = 0; i <= words.length; i++) {
    if (i === words.length || words[i] === ' ') {
      reverseSingleWord(words, firstLetter, i - 1)
      firstLetter = i + 1
    }
  }
}

// can be done with STACK as well, read in and then out, 2n
function reverseSingleWord(words, start, end) {
  while (start < end) {
    let temp = words[start]
    words[start] = words[end]
    words[end] = temp
    start += 1
    end -= 1
  }
}

// reverseWords(message);
// console.log(message.join(''));


const myArray = [3, 4, 6, 10, 11, 15];
const alicesArray = [1, 5, 8, 12, 14, 19];

function mergeArrays(arr1, arr2) {
  const mergedArray = []
  while (arr1.length !== 0 && arr2.length !== 0) {
    if (arr1[0] < arr2[0]) {
      mergedArray.push(arr1.shift())
    } else {
      mergedArray.push(arr2.shift())
    }
  }
  mergedArray.concat(arr1)
  mergedArray.concat(arr2)
  return mergedArray
}

// console.log(mergeArrays(myArray, alicesArray));
// logs [1, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 19]

function isRiffle(shuffledDeck, deck1, deck2) {
  let deck1Index = 0;
  let deck2Index = 0;
  for (let i = 0; i < shuffled.length; i++) {
    if (shuffledDeck[i] && shuffledDeck[i] === deck1[deck1Index]) {
      deck1Index += 1
    }
    else if (shuffledDeck[i] && shuffledDeck[i] === deck2[deck2Index]) {
      deck2Index += 1
    } else {
      return false
    }
  }
  return true
}

function twoSum(arr, target) {
  const seenMovies = new Set()
  for (let i = 0; i < arr.length; i++) {
    if (seenMovies.has(arr[i] - target)) {
      return true;
    }
    seenMovies.add(arr[i])
  }
  return false
}

function permPalindrome(word) {
  const oddLetters = new Set()
  word.forEach(letter => {
    if (oddLetters.has(letter)) {
      oddLetters.delete(letter)
    } else {
      oddLetters.add(letter)
    }
  })
  return oddLetters.size < 2
}

function countWords(words) {
  const wordCounts = new Map();
  words = words.split
  words.forEach(word => {
    // word = remove punctuation from front and back ( ) . ? ! ,
    let val = wordCounts.get(word.toLowerCase) ? wordCounts.get(word.toLowerCase) + 1 : 1
    wordCounts.set(word.toLowerCase, val)
  })
}

const stockPrices = [10, 7, 5, 8, 11, 4, 9];
const stockPrices2 = [5, 4, 3, 2, 1]

function stockPicker(stockPrices) {
  if (stockPrices.length === 1 || !stockPrices) return;
  let maxProfitSoFar = stockPrices[1] - stockPrices[0]
  let lowestPriceSoFar = stockPrices[0]
  for (let i = 1; i < stockPrices.length; i++) {
    let price = stockPrices[i]
    if ((price - lowestPriceSoFar) > maxProfitSoFar) {
      maxProfitSoFar = price - lowestPriceSoFar
    }
    if (lowestPriceSoFar > price) {
      lowestPriceSoFar = price
    }
  }
  return maxProfitSoFar
}

// Find the max product of any 3 ints in an array
// the slowest way (n^3): 
// try every combination
// better (n log n):
// sort the array and then return the product of the last 3 ints 
// fastest (n):
// greedy approach. start with the first 3 numbers, as you loop take a number
// if it is larger than the your smallest of the 3 
// basicly this is a single pass rolling max but on 3 rather than 1 number
// keeping our set of three sorted at each step would be a constant time sort since its
// guranteed to always be 3 numbers 

// to handle negatives we can use the logic of not just checking size but actually get the max
// of the three combos which would be 


const nums = [-10, 1, 2, 3, 4, -10]
const num2 = [1, 2, 3, 4]
const num3 = [722,634,-504,-379,163,-613,-842,-578,750,951,-158,30,-238,-392,-487,-797,-157,-374,999,-5,-521,-879,-858,382,626,803,-347,903,-205,57,-342,186,-736,17,83,726,-960,343,-984,937,-758,-122,577,-595,-544,-559,903,-183,192,825,368,-674,57,-959,884,29,-681,-339,582,969,-95,-455,-275,205,-548,79,258,35,233,203,20,-936,878,-868,-458,-882,867,-664,-892,-687,322,844,-745,447,-909,-586,69,-88,88,445,-553,-666,130,-640,-918,-7,-420,-368,250,-786]
function highestProduct(arr) {
  let highestProductOf3 = arr[0] * arr[1] * arr[2]
  let highestProductOf2 = arr[0] * arr[1]
  let lowestProductOf2 = arr[0] * arr[1]
  let low = Math.min(arr[0], arr[1])
  let high = Math.max(arr[0], arr[1])
  for (let i = 2; i < arr.length; i++) {
    highestProductOf3 = Math.max(
      highestProductOf3,
      highestProductOf2 * arr[i],
      lowestProductOf2 * arr[i]
    )
    highestProductOf2 = Math.max(
      highestProductOf2, 
      arr[i] * high,
      arr[i] * low
    )
    lowestProductOf2 = Math.min(
      lowestProductOf2, 
      arr[i] * low,
      arr[i] * high
    )
    high = Math.max(high, arr[i])
    low = Math.min(low, arr[i])
  }
  return highestProductOf3
}

// console.log(highestProduct(num2))
// console.log(highestProduct(num3))


// You have an array of integers, and for each index you want to find the
// product of every integer except the integer at that index.
function prodExceptSelf(arr) {
  const backwardsProd = arr.slice()
  for (let i = arr.length - 2; i >= 0; i--) {
    backwardsProd[i] = backwardsProd[i] * backwardsProd[i + 1]
  }
  for (let i = 1; i < arr.length; i++) {
    arr[i] = arr[i] * arr[i - 1]
  }
  const answer = []
  answer.push(backwardsProd[1])
  for (let i = 1; i < arr.length - 1; i++) {
    answer.push(arr[i - 1] * backwardsProd[i + 1])
  }
  answer.push(arr[arr.length-2])
  return answer
}

var subsets = function(nums) {
  if (nums === [] || !nums) {
      return [[]]
  }
  if (nums.length === 1) {
      return [nums]
  }
  let first = nums[0];
  let rest = nums.slice(1)
  let result = subsets(rest)
  for (let i = 0; i < result.length; i++) {
    result[i].push(first)
  }
  return result;
};

const aDupe = [1, 3, 4, 1, 2]
const bDupe = [2, 1, 3, 4, 4]
const cDupe = [1, 2, 1]

// if a number ever directed us somewhere and we ended up back there, it must be because we hit the same numnber again. 
// keep track of the last two number's we saw 
// go to whatever index the number were on takes us to 
// if the number we saw 2 numbers ago == current, the last number we saw must be a dupe 

function findDupe(arr) {
  let lastNumber; 
  let twoNumbersAgo;
  let idx = 0;
  while(true) {
    if (twoNumbersAgo === arr[idx]) {
      return lastNumber;
    }
    twoNumbersAgo = lastNumber;
    lastNumber = arr[idx];
    idx = arr[idx]
  }
}

function isValidBST(node) {
  if (!node) return true;
  let rightBalanced = true;
  let leftBalanced = true;
  if (node.left) {
    rightBalanced = node.left.value <= node.value && isValidBST(node.left)
  }
  if (node.right) {
    leftBalanced = node.right.value >= node.value && isValidBST(node.right)
  }
  return rightBalanced && leftBalanced;
}


// choose a number from range 0 - length - 1, and swap that number from the front of the array 
// increase our starting point by 1 

// this is the Fisher-Yates shuffle 

// when we pick A for the first slot, that has a 1/N chance of happening. 
// lets say we pick A for the second slot, the chances of the happening would be 
// the chance of any other number being selected first: (n - 1) / N 
// follow by (multiplied by) the chance of A then being selected: (1 / (n - 1))
// so we get ((n - 1) / N)  * (1 / (n - 1))
// this simplifies to 1/N 
// So each number, no matter when its choosen, has the same 1/N chance of ending up in any slot
// This makes it a even, truly random shuffle. 

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min; //The maximum is exclusive and the minimum is inclusive
}

function inPlaceShuffle(arr) { 
  let start = 0 
  let end = arr.length
  while (start < end) {
    let rand = getRandomInt(start, end);
    [arr[start], arr[rand]] = [arr[rand], arr[start]]
    start++;
  }
}

const randArr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// inPlaceShuffle(randArr)
// console.log(randArr)

function reverseLinkedList(node) {
  let prevNode = null;
  while (node) {
    [node.next, prevNode, node] = [prevNode, node, node.next]
  }
  return prevNode;
}

/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} m
 * @param {number} n
 * @return {ListNode}
 */
var reverseBetween = function(head, m, n) {
  let currentPos = 1; 
  let current = head; 
  let prevNode;
  let beginSwapNode;
  let endSwapNode;
  while (current) {
      if (currentPos === (m - 1)) {
          beginSwapNode = current; 
      }
      if (currentPos === m) {
          endSwapNode = current; 
      }
      if (currentPos === n) {
          beginSwapNode.next = current;
          endSwapNode.next = current.next;
      }
      if (currentPos >= m && currentPos <= n) {
          [current.next, prevNode, current] = [prevNode, current, current.next]
      } else {
          current = current.next;  
      }
      currentPos += 1; 
  }
  return head; 
};

// find the middle point of a rotated array, as determined by alphabetical ordering 
// also consider 

// should return 2 as the point of rotation. 
const numbers = [4, 5, 1, 2, 3] 

const words = [
  'ptolemaic',
  'retrograde',
  'supplant',
  'undulate',
  'xenoepist',
  'asymptote',  // <-- rotates here!
  'babka',
  'banoffee',
  'engender',
  'karpatka',
  'othellolagkage',
];

// if youre going left or right and hit the end, loop to the other side 
// if you move left to find a smaller or right to find a bigger item, and its not, then youve gone too far and 
// you should go back and step half as far. 

// compare current item to first item. if its smaller than first item, go left. if its larger, right. 

function findStart(arr) {
  let start = 0;
  let end = arr.length - 1;
  let mid = end / 2;
  while (mid !== 0 && arr[mid] > arr[mid - 1]) {
    if (arr[mid] < arr[0]) {
      end = mid;
    } else {
      start = mid;
    }
    mid = end / 2;
  }
  return mid;
}

console.log(findStart(words))

// find the first duplicate in an array with numbers from 1 to n and a lenth of n + 1 without
// using extra space

function findDupe(arr) {

} 


// you have a string and number, return the length of the longests sequence of the same letter with up to X replacements 


function findLongestSeq(str) {
  for (let i; i < str.length; i++) {
    let cur = str[i];

  }
}


// delete node from a singlely linked list given only a pointer to that node 

// traverse LL until x.next is target node, then set x's next to targets .next 
// we can be faster, but besides we dont even have a ref to the start!

// we just change the current node to its next's value and then delete it's next

function deleteNode(target) {
  if (target.next) {
    target.val = target.next.val;
    target.next = target.next.next;
  } else {
    throw new Error('Cant delete the last node in a SLL');
  }
}

// see if a singley linked list has a cycle 
// slow runner fast runner, slow takes 1 step at as time, fast takes 2

function containsCycle(head) {
  let slow = head;
  let fast = head;
  while (fast && fast.next) {
    if (slow === fast) {
      return true;
    } 
    slow = slow.next;
    fast = fast.next.next;
  }
  return false; 
}


// return the kth to last node of a singly LL

class LinkedListNode {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

const aN = new LinkedListNode('Angel Food');
const bN = new LinkedListNode('Bundt');
const cN = new LinkedListNode('Cheese');
const dN = new LinkedListNode("Devil's Food");
const eN = new LinkedListNode('Eccles');

aN.next = bN;
bN.next = cN;
cN.next = dN;
dN.next = eN;

console.log(kthToLastNode(2, aN));

function kthToLastNode(k, node) {
  let length = 0; 
  let head = node; 
  while (head) {
    length += 1;
    head = head.next;
  }
  let stoppingPoint = length - k;
  if (stoppingPoint < 1) return null;
  for (let current = 0; current === stoppingPoint; current++) {
    node = node.next;
  }
  return node;
}

// find dup in an array thats n + 1 length and values are 1 .. n 
// do this in n log n time, not the graph theory way

// our dupe could then be (at least) one of 1 .. N
// it must then be in either in the range of (1 .. n/2) or (n/2 .. n)

// to test which range its in 
// we can go through and count how many numbers are in which part of the range 
// this is N 
// we will need to do it log N times 
// n log n runtime 

// if a range 1 ... n/2 has a dupe then it will have one more number 
// than whatever its ending range is, letting us know that this range 
// contains the dupe




function findDupeInArray(arr) {

}

function checkRangeForDupe(arr, from, to) {
  count = 0;
  arr.forEach(num => {
    if (num >= from && num <= to) {
      count += 1;
    }
  })
  return count === (to - from);
}


// leetcode 905 sort things into even then odd order
var sortArrayByParity = function(A) {
  const even = []
  const odd = []
  A.forEach(each => {
      if (each % 2 === 0) {
          even.push(each)
      } else {
          odd.push(each)
      }
  })
  return even.concat(odd);
};

// leetcode 832, reverse then invert all subarrays.
var flipAndInvertImage = function(A) {
  A.forEach(subArr => {
      subArr.reverse()
      for (let i = 0; i < subArr.length; i++) {
          subArr[i] = subArr[i] ^ 1;
      }
  })
  return A
};

var hammingDistance = function(x, y) {
  let num = x ^ y;
  let distance = 0;
  while (num > 0) {
      distance += num & 1
      num = num >>> 1
  }
  return distance;
};

// basic memoized fibonacci 
const fib = function(n) {
  return helper(n, {})
};

function helper(n, mem) {
  if (n === 0) return 0;
  if (n === 1) return 1;
  if (mem[n]) return mem[n];
  mem[n] = helper(n - 1, mem) + helper(n - 2, mem);
  return mem[n];
}


// leetcode 581 shortest unsorted continuous subarray! 
/**
 * @param {number[]} nums
 * @return {number}
 */
var findUnsortedSubarray = function(nums) {
  if (nums.length === 1) return 0;
  let firstUnordered = 0; 
  let lastUnordered = nums.length - 1; 
  for (let i = 0; i < nums.length; i++) {
      if (nums[i] <= nums[i + 1]) {
          firstUnordered++;   
      } else {
          break;
      }
  }
  for (let i = nums.length - 1; i >= 0; i--) {
      if (nums[i] >= nums[i - 1]) {
          lastUnordered--;   
      } else {
          break;
      }
  }
  
  let min = Math.min(...nums.slice(firstUnordered, lastUnordered + 1))
  let max = Math.max(...nums.slice(firstUnordered, lastUnordered + 1))
  
  for (let i = 0; i < firstUnordered; i++) {
      if (min < nums[i]) {
          console.log('setting new front')
          firstUnordered = i;   
      }
  }
  
  for (let i = nums.length - 1; i > lastUnordered; i--) {
      if (max > nums[i]) {
          console.log('setting new end')
          lastUnordered = i;   
      } 
  }
  
  if (lastUnordered === 0 && firstUnordered === 0) return 0;
  let answer = (lastUnordered - firstUnordered) + 1;
  if (answer < 0) return 0;
  return answer;
};

// find the start of the unordering going forward
// find the start of the unordering going backwards
// any single sub sort to order the entire thing must cover minium
// from the start to the end found this way 

// get the min and max from this sorted segment
// if the max is larger than the number right in front of the end of our segment, walk
// it forward until we encompass it
// same thing for the smallest! 


// find first and last unordered thing. 
// find min and max from that segment 



// leet 160 Linked List intersection 

/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */

/**
 * @param {ListNode} headA
 * @param {ListNode} headB
 * @return {ListNode}
 */
var getIntersectionNode = function(headA, headB) {
  let aLength = 0;
  let bLength = 0; 
  let a = headA;
  let b = headB;
  
  while(a) {
      aLength += 1
      a = a.next
  }
  while(b) {
      bLength += 1
      b = b.next
  }

  
  let offset; 
  if (aLength > bLength) {
      offset = aLength - bLength; 
      for (let i = offset; offset > 0; offset--) {
          headA = headA.next;
      }
  }
  if (aLength < bLength) {
      offset = bLength - aLength;
      for (let i = offset; offset > 0; offset--) {
          headB = headB.next;
      }
  }
  
  while(headA && headB) {
      if (headA === headB) {
          return headA
      }
      headA = headA.next;
      headB = headB.next;
  }
};


// leetcode 70 memoized steps 

const mem = {}

var climbStairs = function(n) {
  return helper(n, {})
};

function helper(n) {
  if (n === 0) return 0; 
  if (n === 1) return 1; 
  if (n === 2) return 2; 
  if (mem[n]) return mem[n]
  mem[n] = helper(n - 1) + helper(n - 2)
  return mem[n]
}

function dynamicStairs(n) {
  if (n === 1) return 1;
  if (n === 2) return 2; 
  let arr = [1, 2];
  for (let i = 2; i < n; i++) {
    arr.push(arr[i - 1] + arr[i - 2]);
  }
  return arr[arr.length - 1];
}

// leet code 56 merge intervals  
var mergeIntervals = function(intervals) {
  if (intervals.length === 0) return [];
  intervals.sort((a, b) => a[0] - b[0])
  let newIntervals = [intervals[0]];
  let start; 
  let end;
  for (let i = 1; i < intervals.length; i++) {
      start = intervals[i][0];
      end = intervals[i][1];
      let curEnd = newIntervals[newIntervals.length - 1][1];
      if (start <= curEnd) {
          if (end > curEnd) {
              newIntervals[newIntervals.length - 1][1] = end;
          }
      } else {
          newIntervals.push([start, end]);
      }     
  }
  return newIntervals;
};


//leet code 104 max depth binary tree 

var maxDepth = function(root) {
  if (root === null) return 0;
  return helper(root, 1);
};

function helper(root, count) {
  if (root.left !== null && root.right !== null)  {
      return Math.max(helper(root.right, count + 1), helper(root.left, count + 1))
  }
  if (root.right !== null)  {
      return helper(root.right, count + 1)
  }
  if (root.left !== null) {
      return helper(root.left, count + 1)
  }
  return count;
}

//leet code valid parenths 
var isValid = function(s) {
  let paren = [];
  for (let i = 0; i < s.length; i++) {
      let cur = s[i]
      if (cur === '(' || cur === '{' || cur === '[') {
          paren.push(cur);
      } else {
          let temp = paren.pop()
          if (!temp) return false;
          temp = temp + cur
          if (temp !== '()' && temp !== '{}' && temp !== '[]') {
              return false; 
          } 
      }
  }
  if (paren.length > 0) return false;
  return true; 
};


// max width sliding window problem 
// max sum of a window of width W 
const holes = [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0]; 

function whackAMole(holes, width) {
  let i = 0; 
  let currentWindow = holes.slice(0, width).reduce((a, b) => a + b, 0); 
  let max = currentWindow;
  if (holes.length < width) return holes.reduce((a, b) => a + b, 0)
  for (let k = width; k < holes.length; k++) {
    currentWindow = currentWindow - holes[i] + holes[k];
    if (currentWindow > max) {
      max = currentWindow;
    }
    i++;
  }
  return max; 
}

console.log('whack a mole')
console.log(whackAMole(holes, 5));

var moveZeroes = function(nums) {
  let count = 0;
  while (nums.indexOf(0) > -1) {
    nums.splice(nums.indexOf(0), 1)
    count += 1;
  }
  nums.push(... new Array(count).fill(0));
};

const moveZ1 = [0,0,1]

var moveZeroes2 = function(nums) {
  let lastNonZeroFoundAt = 0; 
  let temp;
  for (let cur = 0; cur < nums.length; cur++) {
      if (nums[cur] !== 0) {
          temp = nums[lastNonZeroFoundAt];
          nums[lastNonZeroFoundAt] = nums[cur];
          nums[cur] = temp;
          lastNonZeroFoundAt += 1;
      }
  }
};

console.log('move zeros')
moveZeroes(moveZ1)
console.log(moveZ1)

var deleteNode = function(node) {
  node.val = node.next.val
  node.next = node.next.next
};


//sorting 
var majorityElement = function(nums) {
  if (nums.length === 1 || nums.length === 2) return nums[0]
  nums.sort((a, b) => a - b)
  return nums[((nums.length - 1)/2) >> 0]
};

//voting with boyer moore
var majorityElement = function(nums) {
  let count = 0;
  let candidate;
  for (let i = 0; i < nums.length; i++) {
      if (count === 0) {
          candidate = nums[i];
      }
      if (nums[i] === candidate) {
          count += 1;
      } else {
          count -= 1;
      }
  }
  return candidate;
}


class MaxStack {
  constructor() {

    // Initialize an empty stack
    this.items = [];
    this.max = [];
  }

  // Push a new item onto the stack
  push(item) {
    this.items.push(x)
    if (this.max.length === 0 || this.max[this.max.length - 1] >= x) {
      this.max.push(x);
    }
  }

  getMax() {
    return this.max[this.max.length - 1] ;
  }

  // Remove and return the last item
  pop() {
    const last = this.items.pop();
    if (last === this.max[this.max.length - 1]) {
      this.max.pop();
    }
    return last
  }

  // Returns the last item without removing it
  peek() {
    if (!this.items.length) {
      return null;
    }
    return this.items[this.items.length - 1];
  }
}


// KNAP SACK BABY 
const menu = {
  "Fruit": 215,
  "Fries": 275,
  "Salad": 335,
  "Wings": 355,
  "Mozzarella": 420,
  "Plate": 580
}

function tryAllKnapsack(budget, menu, order, solutions) {
  if (budget === 0) { solutions.push(order); return }
  for (const [item, cost] of Object.entries(menu)) {
    if (cost <= budget) {
      let currentOrder = [...order];
      currentOrder.push(item);
      tryAllKnapsack(budget - cost, menu, currentOrder, solutions)
    }
  }
}

function tryAllKnapSackStarter(budget, menu) {
  const solutions = [];
  tryAllKnapsack(budget, menu, [], solutions)
  return solutions;
}

console.log(tryAllKnapSackStarter(1505, menu));


const numberTranslations =  {
  0: 'zero',
  1: 'one',
  2: 'two',
  3: 'three',
  4: 'four',
  5: 'five',
  6: 'six',
  7: 'seven',
  8: 'eight',
  9: 'nine',
  10: 'ten',
  11: 'eleven',
  12: 'twelve',
  13: 'thirteen',
  14: 'fourteen',
  15: 'fifteen',
  16: 'sixteen',
  17: 'seventeen',
  18: 'eighteen',
  19: 'nineteen',
  20: 'twenty',
  30: 'thirty',
  40: 'fourty',
  50: 'fifty',
  60: 'sixty',
  70: 'seventy',
  80: 'eighty',
  90: 'ninety'
}

const magnitudes = ['', ' thousand ', ' million ', ' billion ', ' trillion '];

// 7 - > seven 
// 30 -> thirty 
// 32 -> thirty two
// 123 -> one hundred twenty three 
// 723 -> seven hundred twenty three
// 7001 -> seven thousand one
// 7023 -> seven thousand twenty three 
// 74023 -> seventy four thousand twenty three

// convert blocks of 0-99 then x00, and then lace magnitude marker after each block 

// 0-19 direct conversion, special 
// xx - convert the first then the second, ignore zeros
// xxx - convert the first, add marker, then the next 2 
// 

// 990 nine hundred ninety  
// 9/99/  THOUSAND , 1/00 nintey nine hundred 

function translateNumToString(num) {
  let result = '';
  for (let i = 0; num > 0; i++) {
    if (i >= magnitudes.length) throw new Error('Number is too large to convert.');
    result = (convertBlock(num % 1000) + magnitudes[i]) + result;
    num = Math.floor(num/1000);
  }
  return result; 
}

// convert numbers 0 - 999
function convertBlock(num) {
  let result = '';
  // handle 0-20
  if (num <= 20) return numberTranslations[num];
  // handle hundreds if applicable
  if (num >= 100) {
    let hundredsDigit = Math.floor(num / 100)
    result += numberTranslations[hundredsDigit] + ' hundred';
  }
  // handle first digit of 21-99
  let tensDigit = Math.floor(num / 10) % 10
  if (tensDigit !== 0) result += ' ' + numberTranslations[tensDigit * 10];
  // handle last digit
  if (num % 10 !== 0) result += ' ' + numberTranslations[num % 10];
  return result; 
}

console.log(convertBlock(829));
console.log(convertBlock(0));
console.log(convertBlock(17));
console.log(convertBlock(127));
console.log(convertBlock(999));
console.log(translateNumToString(383));
console.log(translateNumToString(1383));
console.log(translateNumToString(31383));
console.log(translateNumToString(13564));
console.log(translateNumToString(1883564));


// 841 leetcode 
var canVisitAllRooms = function(rooms) {
  let seen = new Set();
  let toVisit = [0];
  while (toVisit.length > 0) {
      let key = toVisit.pop()
      seen.add(key)
      rooms[key].forEach(e => { if (!seen.has(e)) toVisit.push(e) });
  }
  return seen.size === rooms.length;
};  


// Given a hallway with a set of lights in various positions and of various light casting radii,
// determine if the hallway can be crossed while remaining in the unlit areas of the hallway.
// ** OR **
//Given the radius / x and y coordinates of the middle of a list of circles, and the height of the 
// y axis, determine if a grid is traversible on the x axis by a 1x1 square.


// Print out a tree's paths from root to leaves 

// find k top values from a data set

// Compare a hash table by key and by value, including recursively flattening nested hash tables. 

// 

// priority q, with comparator 

// Given an unsigned value, implement the two's complement value.  

var lengthOfLongestSubstring = function(s) {
  let max = 0;
  let set = new Set(); 
  let current = 0;
  while (current < s.length) {
      for (let i = current; i < s.length; i++) {
          if (set.has(s[i])) {
              current += 1;
              set.clear();
              break;
          } else {
              set.add(s[i]);
              max = Math.max(set.size, max);
          }
      }
  }
  return max; 
};


// leet code 17 

const phoneetters = {
  '2': 'abc',
  '3': 'def',
  '4': 'ghi',
  '5': 'jkl',
  '6': 'mno',
  '7': 'pqrs',
  '8': 'tuv',
  '9': 'wxyz'
}

var letterCombinations = function(digits) {
  if (digits.length === 1 ) return phoneletters[digits].split('');
  if (digits === '') return [];
  let result = [];
  let cur = digits[0];
  digits = digits.slice(1);
  for (let i = 0; i < phoneletters[cur].length; i++) {
      let letter = phoneletters[cur][i];
      let previous = letterCombinations(digits);
      previous.forEach(combo => {
          result.push(letter + combo);
      })
  }
  return result; 
};


function allPerms(str, set = []) {
  if (str.length === 1) return [str];
  for (let i = 0; i < str.length; i++) {
    let cur = str[i];
    let result = allPerms(str.replace(cur, ''));
    set = set.concat(result.map(each => cur + each))
  }
  return set;
}

console.log('perms')
console.log(allPerms('ab'))
console.log(allPerms('abc'))


const cakeTypes = [
  { weight: 7, value: 160 },
  { weight: 3, value: 90 },
  { weight: 2, value: 15 },
];

const capacity = 20;

function maxDuffelBagValue(cakeTypes, capacity, mem = {}) {
  const maxValuesAtCapacities = new Array(capacity + 1).fill(0);
  for (let curCapacity = 1; curCapacity <= capacity; i++) {
    cakeTypes.forEach(cakeType => {
      if (cakeType.weight === 0 && cakeType.value > 0) {
        // return Infinity
      }
      if (cakeType.weight <= curCapacity) {
        takeCurrentCake = maxValuesAtCapacities[curCapcity - cakeType.weight] + cakeType.value;
        mem[curCapcity] = Math.max(mem[curCapcity], takeCurrentCake)
      }
    })
  }
  return maxValuesAtCapacities[capacity]; 
}

maxDuffelBagValue 


// "Of course, I'd love to get the job. But whether you hire me, or someone else:
//  when you think about the future, a year from now, how will you know whether or not 
//  you hired the right person? What will that person have to have done, for you to know 
//  that they were the right choice?"

var LCmissingNumber = function(nums) {
  let expectedSum = (nums.length * (nums.length + 1)) / 2;
  let actualSum = 0;
  nums.forEach(each => actualSum += each)
  return expectedSum - actualSum;
};


var rotatedBS = function(nums, target) {
  let mid = Math.floor(nums.length / 2);
  let left = 0;
  let right = nums.length -1; 
  while(left <= right) {
      console.log(nums[mid])
      if (nums[mid] === target) return mid;
      if (true) {
          left = mid + 1; 
      } else {
          right = mid - 1; 
      }
      mid = Math.floor((right + left) / 2);
  }
  return -1;
};

var firstMissingPositive = function(nums) {
    for (let i = 0; i < nums.length; i++) {
      recSwap(nums, nums[i]) 
    }
    for (let i = 0; i < nums.length; i++) {
      if (nums[i] !== i + 1) return i + 1; 
    }
    return nums.length; 
};

function recSwap(nums, numToMove) {
  let canIgnore = numToMove > 0 || numToMove > nums.length;
  let alreadyThere = numToMove === nums[num - 1]
  if (canIgnore || alreadyThere) return;
  const displacedNum = nums[numToMove - 1];
  nums[numToMove - 1] = numToMove;
  recSwap(nums, displacedNum);
}

console.log('first missing positive')
console.log(firstMissingPositive([3, 4, -1, 1]))
console.log(firstMissingPositive([7, 8, 9, 11, 12]))
console.log(firstMissingPositive([1, 2, 0]))

// if the num is > the array size then we can ignore it
// can ignore negatives 
// can ignore duplicates 
// can ignore 0 

// place each item in its place in the array 
// go over it again and return first non-seqenutial hit 

// if you replace a number that isnt to be ignored,
// then place it and if you replace a number that shouldnt be ingored, etc

// *iter
// [3 2 1]
// [3 2 3] need to place 1
// [1 2 3] need to place 3 
// [1 2 3] 3 already there so stop 
// *iter
// [1 2 3] 2 already there so stop
// *iter
// [1 2 3] 3 already there so stop





