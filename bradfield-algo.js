// A pangram is a phrase which contains every letter at least once, 
// such as “the quick brown fox jumps over the lazy dog”. 
// Write a function which determines if a given string is a pangram.

// A pangram contains every letter from a given set ( the alphabet ),
// so we can keep track of all these letters and mark them off when we see them. 
// If we see them all, then we know it's a pangram, but if we reach the 
// end of our string without seeing them all then it's not. 

// Make set out of the alphabet.
// Loop over string. 
// Remove each letter from the set as we see it if it's in the set.
// Check if the set is empty. 
// If we reach the end of the loop without an empty set, its not a pangram.

function isPangram(str) {
  let set = new Set([...Array(26)].reduce(a=>a+String.fromCharCode(i++),'',i=97));
  for(let i =0; i < str.length; i++) {
    if (set.has(str[i])) set.delete(str[i]);
    if (set.size === 0) return true;
  }
  return false;
}

// Count each letter in our string and at the end if there is at least 
// 1 of every letter then it must be a pangram. this assumes only valid
// letters will be in the string.

function isPangram2(str) {
  let count = {}
  for(let i =0; i < str.length; i++) {
    if (count[str[i]]) { 
      count[str[i]] += 1;
    } else {
      count[str[i]] = 1;
    }
  }
  return Object.keys(count).length === 26;
}

// Take the letters we know must be there and then search for  
// each of them in the string 

function isPangram3(str) {
  let alpha = 'abcdefghijklmnopqrstuvwxyz';
  for (let i = 0; i < alpha.length; i++) {
    if (str.indexOf(alpha[i]) === -1) return false;
  }
  return true;
}



// Given a width and a height make a 2D array with a spiral pattern. 
// "How do you go right?"" what does it mean to go right? 
// left, down, up, etc

// we can keep track of a current posistion cursor (row, col) 
// going right would mean incrementing row 
// going down means incrementing col 

// "How do I know when to stop?"
// "When do I turn?"
// "How do I turn?"
// "How do I know what my current direction is"





// Write a function that uses a stack (you can use the stack implementation in /algos) 
// to return a reversed copy of a list.

// Well, the call stack is a stack so... :P
function reverseListUsingStack(lst) {
  if (lst.length === 0) return lst;
  let rest = lst.slice(1);
  return reverseListUsingStack(rest).concat(lst[0])
}
console.log(reverseListUsingStack([1, 2, 3]))

function reverseListUsingStack2(lst) {
  let [stack, result] = [[],[]];
  lst.forEach(item => {
    stack.push(item);
  })
  for (let i = 0; i < lst.length; i++) {
    result.push(stack.pop())
  }
  return result;
}

console.log(reverseListUsingStack2([1, 2, 3]))

// Implement a queue using stacks.Although this problem might seem contrived, 
// implementing a queue using stacks is actually a common strategy 
// in functional programming languages (such as Haskell or Standard ML) 
// where stacks are much more convenient to implement “from scratch” than queues. 
// Your solution should have the same interface as the queue implementation in algos, 
// but use stacks instead of a list in the implementation.

class Queue {
  constructor() {
      this._items = []
      this._flippedItems = []
  }
  is_empty() {
      return this._items === [] && this._flippedItems === [];
  }
  enqueue(item) {
      this._items.push(item)
  }
  dequeue() {
    if (this._flippedItems.length === 0) {   
      this._flip();
    }
    return this._flippedItems.pop()
  }
  size() {
      return this._items.length + this._flippedItems.length;
  }
  _flip() {
    while(this._items.length > 0) {
      this._flippedItems.push(this._items.pop());
    }
  }
}

let q = new Queue();
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
console.log(q.dequeue())
q.enqueue(4)
q.enqueue(5)
console.log(q.dequeue())
console.log(q.dequeue())
q.enqueue(6)
console.log(q.dequeue())
console.log(q.dequeue())
console.log(q.dequeue())
q.enqueue(7)
q.enqueue(8)
console.log(q.dequeue())


// file paths 
// given a path 
// path can contain .. to mean go up 
// . means in current directory 
// simplify out the single and double dots 

function parsePath(path) {
  let stack = []; 
  path = path.split('/');
  path.forEach(each => {
    if (each === '..') {
      stack.pop()
      stack.pop()
    } else if (each !== '.') {
      stack.push('/')
      stack.push(each);
    }
  })
  return stack.join('');
}

const path = 'usr/elliot/foo/.././bar';
console.log(parsePath(path));






