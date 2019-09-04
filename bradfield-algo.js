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


