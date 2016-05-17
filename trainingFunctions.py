import math
import numpy as np

def evenParity(n):
	"""Determines if n is an even parity.

	Calculates the number of 1 bits in the binary representation of the given 
	integer, n. An even parity is true if the number of 1 bits in the
	representation is even.

	Args:
		n: integer to check for even parity.

	Returns:
		A 0 if n is an even parity, else a 1.
	"""
	binary =  "{0:b}".format(n)
	return binary.count('1') % 2

def oddParity(n):
	"""Determines if n is an odd parity.

	Calculates the number of 1 bits in the binary representation of the given 
	integer, n. If the number is odd,  outputs a 0. If the number is even, 
	outputs a 1.

	Args:
		n: integer to check for odd parity.

	Returns:
		A 0 if n is an odd parity, else a 1.
	"""
	binary = "{0:b}".format(n)
	return (binary.count('1') + 1) % 2

def adder(n):
	"""Adds 42 to a given number.

	Args:
		n: integer to add to.

	Returns:
		The value of n added by 42.
	"""
	return n + 42

def addThem(n, m):
	"""Calculates the sum of n and m.

	Args:
		n: an integer to add.
		m: an integer to add.
	Returns:
		The sum of n and m.
	"""
	return n + m

def multiply(n, m):
	"""Returns n multiplied by m.

	Args:
		n: integer to be multiplied by
		m: integer to be multiplied by

	Returns:
		The product of n and m.
	"""
	return n * m

def multiplyAndAdd(n, m):
	"""Adds 5 to n, adds 7 to m, and returns the product of the two.

	f(n, m) = (n + 5) * (m + 7)

	Args:
		n: integer as an input to the function
		m: integer as an input to the function

	Returns:
		The output of the function given n and m.
	"""
	return (n + 5) * (m + 7)

def makePalindrome(s):
	"""Makes a palindrome from a given string.

	Reverses and appends s to the end of s. This guarantees the return 
	statement is a palindrome.

	Args:
		s: a string

	Returns:
		A palindrome formed by appending s reversed to the end of s.
	"""
	return s + s[::-1]

def isPalindrome(s):
	"""Validates whether the input is a palindrome.

	Args:
		s: a string

	Returns:
		True if the input is a valid palindrome, else False.
	"""
	return all(s[i] == s[-i - 1] for i in range(len(s) >> 1))

def sine(x):
	"""Returns the sin of x radians.

	Args:
		x: number of radians

	Returns:
		The sine of x radians.
	"""
	return math.sin(x)

def fib(n):
	"""Approximates the nth Fibonacci number.

	Since this uses Binet's formula, it is only exactly accurate from 1 to 70.
	Past n values of 70, it is only an approximation.

	Binet's formula is defined as:
	F(n) = ((Phi ^ n) - ((- Phi) ^ -n)) / sqrt(5)
			 = (((1 + sqrt(5)) / 2) ^ n) - ((1 - sqrt(5)) / 2) ^ n)) / sqrt(5))

	Args:
		n: integer for the nth number in the Fibonacci sequence

	Returns:
		The output of Binet's forumula with input n, an approximation of the nth
		number in the Fibonacci sequence.
	"""
	return round(((((1 + math.sqrt(5)) / 2) ** n) - (((1 - math.sqrt(5)) / 2) ** n)) / math.sqrt(5))

def testFibAccuracy():
	"""Calculates when the function fib(n) becomes inaccurate.

	Exactly up to 70.
	71 - 72 are off by 1.
	73 and on are off by more than 1.

	Returns:
		fn: the last Fibonacci number found by using the recursive function.
		count: the number of Fibonacci numbers calculated.
		fib(count): an approximation for the last Fibonacci number.
	"""
	fn = f1 = f2 = 1
	count = 2
	while (fn == fib(count)) | (fn == fib(count) + 1) | (fn == fib(count) - 1):
		fn = f1 + f2
		f2, f1 = f1, fn
		count+=1
	return fn, count, fib(count)

def determinant(m):
	"""Calculates the determinant of a matrix.

	Args:
		m: array of arrays that represents a matrix.

	Returns:
		The determinant of a matrix.
	"""
	return np.linalg.det(m)

def main():
	print("evenParity(2) ->", evenParity(2))
	print("evenParity(3) ->", evenParity(3))
	print("evenParity(10) ->", evenParity(10))
	print("oddParity(2) ->", oddParity(2))
	print("oddParity(3) ->", oddParity(3))
	print("oddParity(10) ->", oddParity(10))
	print("oddParity(1243) + evenParity(1243) == 1 ->", 
		   oddParity(1243) + evenParity(1243) == 1)
	print("adder(1) ->", adder(1))
	print("adder(5) ->", adder(5))
	print("multiply(3, 7) ->", multiply(3, 7))
	print("multiplyAndAdd(3, 7) ->", multiplyAndAdd(3, 7))
	print("isPalindrome('abc') ->", isPalindrome('abc'))
	print("isPalindrome('yootnooy') ->", isPalindrome('yootnooy'))
	print("makePalindrome('This is the final countdown!') ->",
		makePalindrome('This is the final countdown!'))
	print("isPalindrome(makePalindrome('This is the final countdown!')) ->", 
		isPalindrome(makePalindrome('This is the final countdown!')))
	print("sine(math.pi) ->", sine(math.pi))
	print("sine(math.pi / 2) ->", sine(math.pi / 2))
	print("fib(2) ->", fib(2))
	print("fib(3) ->", fib(3))
	print("fib(4) ->", fib(4))
	print("fib(5) ->", fib(5))
	print("fib(6) ->", fib(6))
	fn, count, approx = testFibAccuracy()
	print("Actual", count, "number in fib sequence =", fn, 
		"Approximation =", approx)

if __name__ == "__main__":
    main()