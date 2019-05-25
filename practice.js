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























