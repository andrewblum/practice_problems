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

// two linked lists, unknown lengths, merge at some point, return node they meet at 
// do it in constant space, linear time

// one will reach the end first, and you can count how many nodes ahead of the other runner it was
// this is your offset and will let you compare accross to the two ll's and find the merge, this is 2m + 2n time. 

function mergeNode(l1, l2) {

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