#!/usr/bin/env python3
"""
Time Complexity Quiz - Test your understanding of algorithm analysis

This interactive quiz covers Big O notation, time complexity analysis,
and algorithm performance concepts.
"""

import random
import time
import os


class TimeComplexityQuiz:
    def __init__(self):
        self.score = 0
        self.total_questions = 0
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def ask_question(self, question, options, correct_index):
        """Ask a multiple choice question"""
        self.total_questions += 1
        
        print(f"\n{self.total_questions}. {question}")
        for i, option in enumerate(options, 1):
            print(f"   {i}. {option}")
        
        try:
            answer = int(input("\nYour answer (number): "))
            if answer == correct_index + 1:  # +1 because options are 1-indexed for the user
                print("✅ Correct!")
                self.score += 1
                return True
            else:
                print(f"❌ Incorrect! The correct answer is: {options[correct_index]}")
                return False
        except (ValueError, IndexError):
            print("❌ Invalid input! Please enter a number corresponding to an option.")
            return False
    
    def run_quiz(self, questions, num_questions=10):
        """Run through quiz questions"""
        self.clear_screen()
        print("\n" + "=" * 60)
        print("📊 TIME COMPLEXITY QUIZ - Test Your Algorithm Knowledge")
        print("=" * 60)
        
        # Shuffle questions
        random.shuffle(questions)
        
        # Ask questions
        for q in questions[:num_questions]:
            self.ask_question(q["question"], q["options"], q["correct"])
            time.sleep(0.5)  # Short pause between questions
        
        # Calculate percentage
        percentage = (self.score / self.total_questions) * 100
        
        # Show results
        print("\n" + "=" * 60)
        print(f"📋 Quiz Results: {self.score}/{self.total_questions} correct ({percentage:.1f}%)")
        
        # Give feedback based on score
        if percentage >= 90:
            print("🏆 Outstanding! You have excellent understanding of time complexity!")
        elif percentage >= 70:
            print("🎓 Great job! You have a solid grasp of the concepts.")
        elif percentage >= 50:
            print("📚 Good effort! With a bit more study, you'll master these concepts.")
        else:
            print("📝 Keep studying! Time complexity is a challenging but important topic.")
        
        print("=" * 60)


def generate_basic_questions():
    """Generate a pool of basic quiz questions"""
    return [
        {
            "question": "What is the time complexity of accessing an element in an array by index?",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
            "correct": 0
        },
        {
            "question": "Which sorting algorithm has the best average-case time complexity?",
            "options": ["Bubble Sort", "Insertion Sort", "Quick Sort", "Selection Sort"],
            "correct": 2
        },
        {
            "question": "What is the time complexity of binary search?",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
            "correct": 1
        },
        {
            "question": "Which of the following time complexities is the most efficient for large inputs?",
            "options": ["O(n²)", "O(n log n)", "O(n)", "O(log n)"],
            "correct": 3
        },
        {
            "question": "What is the time complexity of bubble sort in the worst case?",
            "options": ["O(n)", "O(n log n)", "O(n²)", "O(2ⁿ)"],
            "correct": 2
        },
        {
            "question": "In Big O notation, we focus on the:",
            "options": [
                "Best-case scenario",
                "Average-case scenario",
                "Worst-case scenario",
                "All scenarios equally"
            ],
            "correct": 2
        },
        {
            "question": "Which of the following is NOT an example of an O(1) operation?",
            "options": [
                "Accessing an array element by index",
                "Push/pop operations on a stack",
                "Finding the minimum value in an unsorted array",
                "Checking if a hash table contains a key"
            ],
            "correct": 2
        },
        {
            "question": "For large values of n, which is faster?",
            "options": ["O(n log n)", "O(n²)", "They are the same", "It depends on the implementation"],
            "correct": 0
        },
        {
            "question": "What is the time complexity of searching for an element in a balanced binary search tree?",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
            "correct": 1
        },
        {
            "question": "Which data structure typically provides O(1) insertion and removal at one end?",
            "options": ["Array", "Queue", "Stack", "Linked List"],
            "correct": 2
        },
        {
            "question": "What is the space complexity of an algorithm that creates an n×n matrix?",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
            "correct": 3
        },
        {
            "question": "What is the time complexity of the recursive Fibonacci algorithm without memoization?",
            "options": ["O(n)", "O(n log n)", "O(n²)", "O(2ⁿ)"],
            "correct": 3
        },
        {
            "question": "Which of these search algorithms requires a sorted array?",
            "options": ["Linear Search", "Binary Search", "Depth-First Search", "Breadth-First Search"],
            "correct": 1
        },
        {
            "question": "What does 'amortized time complexity' refer to?",
            "options": [
                "The average time complexity over all possible inputs",
                "The time complexity in the worst-case scenario",
                "The average time cost per operation over a sequence of operations",
                "The time complexity when the algorithm terminates early"
            ],
            "correct": 2
        },
        {
            "question": "Which of these is NOT a common technique for improving algorithm efficiency?",
            "options": [
                "Dynamic Programming",
                "Increasing the input size",
                "Greedy Algorithms",
                "Divide and Conquer"
            ],
            "correct": 1
        }
    ]


def generate_code_questions():
    """Generate questions about analyzing code time complexity"""
    return [
        {
            "question": "What is the time complexity of this code?\n\n```python\nfor i in range(n):\n    print(i)\n```",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
            "correct": 2
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\nfor i in range(n):\n    for j in range(n):\n        print(i, j)\n```",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
            "correct": 3
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\ni = n\nwhile i > 0:\n    i = i // 2\n```",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
            "correct": 1
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\nfor i in range(n):\n    j = 1\n    while j < n:\n        j = j * 2\n```",
            "options": ["O(n)", "O(n log n)", "O(log n)", "O(n²)"],
            "correct": 1
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n            \n    return -1\n```",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
            "correct": 1
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\ndef recursive_sum(n):\n    if n <= 0:\n        return 0\n    return n + recursive_sum(n-1)\n```",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
            "correct": 2
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\ndef function(n):\n    if n <= 1:\n        return 1\n    return function(n-1) + function(n-1)\n```",
            "options": ["O(n)", "O(n log n)", "O(n²)", "O(2ⁿ)"],
            "correct": 3
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\ndef find_max(arr):\n    max_val = arr[0]\n    for num in arr:\n        if num > max_val:\n            max_val = num\n    return max_val\n```",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
            "correct": 2
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\ndef find_duplicates(arr):\n    duplicates = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j] and arr[i] not in duplicates:\n                duplicates.append(arr[i])\n    return duplicates\n```",
            "options": ["O(n)", "O(n log n)", "O(n²)", "O(n³)"],
            "correct": 2
        },
        {
            "question": "What is the time complexity of this code?\n\n```python\ndef matrix_multiplication(A, B):\n    n = len(A)\n    C = [[0 for _ in range(n)] for _ in range(n)]\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                C[i][j] += A[i][k] * B[k][j]\n    return C\n```",
            "options": ["O(n)", "O(n²)", "O(n³)", "O(2ⁿ)"],
            "correct": 2
        }
    ]


def generate_advanced_questions():
    """Generate advanced algorithm analysis questions"""
    return [
        {
            "question": "What is the time complexity of Dijkstra's algorithm for finding shortest paths in a graph with V vertices and E edges using a binary heap?",
            "options": ["O(V)", "O(E)", "O(V log V)", "O((V + E) log V)"],
            "correct": 3
        },
        {
            "question": "What is the best time complexity possible for comparison-based sorting algorithms?",
            "options": ["O(n)", "O(n log n)", "O(n²)", "O(1)"],
            "correct": 1
        },
        {
            "question": "Which of the following is NOT a sorting algorithm with O(n log n) average time complexity?",
            "options": ["Merge Sort", "Quick Sort", "Heap Sort", "Insertion Sort"],
            "correct": 3
        },
        {
            "question": "In the context of P vs NP, which statement is true?",
            "options": [
                "All problems in P can be solved in polynomial time",
                "All problems in NP cannot be solved in polynomial time",
                "If P = NP, then all NP-complete problems can be solved in constant time",
                "NP stands for 'not polynomial'"
            ],
            "correct": 0
        },
        {
            "question": "Which data structure would give the best performance for both insertions and lookups in a dictionary-like implementation?",
            "options": ["Array", "Linked List", "Binary Search Tree", "Hash Table"],
            "correct": 3
        },
        {
            "question": "What is the time complexity of the Floyd-Warshall algorithm for finding shortest paths between all pairs of vertices in a graph?",
            "options": ["O(V²)", "O(V³)", "O(V × E)", "O(V × E log V)"],
            "correct": 1
        },
        {
            "question": "Which of these problems is NP-complete?",
            "options": ["Finding shortest path in a graph", "Binary search", "Traveling Salesman Problem", "Merge sort"],
            "correct": 2
        }
    ]


def main():
    """Main quiz execution function"""
    quiz = TimeComplexityQuiz()
    
    # Combine all question types
    all_questions = generate_basic_questions() + generate_code_questions() + generate_advanced_questions()
    
    # Ask for difficulty level
    print("Welcome to the Time Complexity Quiz!")
    print("\nChoose difficulty level:")
    print("1. Beginner (10 questions)")
    print("2. Intermediate (15 questions)")
    print("3. Advanced (20 questions with more difficult topics)")
    
    try:
        level = int(input("\nEnter your choice (1-3): "))
        if level == 1:
            questions = generate_basic_questions()
            num_questions = 10
        elif level == 2:
            questions = generate_basic_questions() + generate_code_questions()[:5]
            num_questions = 15
        elif level == 3:
            questions = all_questions
            num_questions = 20
        else:
            print("Invalid choice, defaulting to Beginner level.")
            questions = generate_basic_questions()
            num_questions = 10
    except ValueError:
        print("Invalid input, defaulting to Beginner level.")
        questions = generate_basic_questions()
        num_questions = 10
    
    # Run the quiz
    quiz.run_quiz(questions, num_questions)
    
    # Ask if user wants to try code analysis section
    try_code = input("\nWould you like to specifically practice code analysis questions? (y/n): ").lower()
    if try_code == 'y':
        print("\n" + "=" * 60)
        print("🧮 CODE ANALYSIS PRACTICE")
        print("=" * 60)
        code_questions = generate_code_questions()
        random.shuffle(code_questions)
        code_quiz = TimeComplexityQuiz()
        code_quiz.run_quiz(code_questions, 5)
    
    print("\nThank you for taking the Time Complexity Quiz!")
    print("Keep learning and practicing algorithm analysis!")


if __name__ == "__main__":
    main()
