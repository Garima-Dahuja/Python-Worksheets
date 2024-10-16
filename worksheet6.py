#ques1
# This project involves analysing the marks of students across multiple subjects. The goal is
# to generate and analyse data, including calculating total marks, average marks, subject-wise
# performance, and identifying top and bottom performers. You will also determine the passing
# percentage and generate insights based on students' performance using plots. 

import matplotlib.pyplot as plt
students = ['Arin', 'Aditya', 'Chirag', 'Gurleen', 'Kunal']
marks = {
    'Math': [85, 79, 90, 66, 70],
    'Physics': [78, 82, 85, 75, 68],
    'Chemistry': [92, 74, 89, 80, 75],
    'English': [88, 90, 92, 78, 85]
}
total_marks = []
average_marks = []
for i in range(len(students)):
    total = marks['Math'][i] + marks['Physics'][i] + marks['Chemistry'][i] + marks['English'][i]
    total_marks.append(total)
    average_marks.append(total / 4)
top_performer = students[total_marks.index(max(total_marks))]
bottom_performer = students[total_marks.index(min(total_marks))]
percentages = [(total / 400) * 100 for total in total_marks]
plt.figure(figsize=(10, 6))
plt.bar(students, percentages, color=['blue', 'green', 'red', 'orange', 'purple'])
plt.xlabel('Student Name')
plt.ylabel('Percentage')
plt.title('Student Performance in Percentage')
plt.show()
print("Total Marks: ", total_marks)
print("Average Marks: ", average_marks)
print(f"Top Performer: {top_performer}")
print(f"Bottom Performer: {bottom_performer}")


#ques2
# You are provided with the following data for the velocity of an object over time. The
# velocity data is modelled by a quadratic function of the form v(t)=at2+bt+c , where v(t) is the
# velocity at time t, and a, b, and c are constants to be determined using curve fitting. (Using
# SciPy's Curve fitting). Plot the original data and curve obtained in one plot with all the
# features of plot.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
time = np.array([0, 1, 2, 3, 4, 5])
velocity = np.array([2, 3.1, 7.9, 18.2, 34.3, 56.2])
def quadratic(t, a, b, c):
    return a * t**2 + b * t + c
params, _ = curve_fit(quadratic, time, velocity)
a, b, c = params
fitted_velocity = quadratic(time, a, b, c)
plt.scatter(time, velocity, label='Original Data', color='red')
plt.plot(time, fitted_velocity, label=f'Fitted Curve: v(t)={a:.2f}t^2 + {b:.2f}t + {c:.2f}', color='blue')
plt.xlabel('Time (Seconds)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time with Curve Fitting')
plt.legend()
plt.show()

#ques3
# You are given the following data representing the population of a town in various years, 
# What is the Pearson's correlation coefficient for the above data? Estimate the population of
# the town in the year 2008 using linear interpolation/regression equation based on Table 1
# data. Write Python code to perform the interpolation using Scipy functions and plot it. 

from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
year = np.array([2000, 2005, 2010, 2015, 2020])
population = np.array([50, 55, 70, 80, 90])
corr, _ = pearsonr(year, population)
print(f"Pearson's correlation coefficient: {corr}")
interp_func = interp1d(year, population, kind='linear')
population_2008 = interp_func(2008)
print(f"Estimated population in 2008: {population_2008}")
plt.scatter(year, population, label='Original Data', color='red')
plt.plot(year, population, label='Population Trend', color='blue')
plt.scatter(2008, population_2008, color='green', label=f'Estimated Population (2008): {population_2008:.2f}')
plt.xlabel('Year')
plt.ylabel('Population (in thousands)')
plt.title('Population Growth and Interpolation')
plt.legend()
plt.show()


#ques4
# Consider the polynomial equation p(x)=3*x^3−5*x^2+2*x−8
# 1. Use SciPy to find the roots of the polynomial.
# 2. Plot the polynomial function p(x) for the range of x from -3 to 3 and mark the roots on
# the plot. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
def p(x):
    return 3*x**3 - 5*x**2 + 2*x - 8
initial_guesses = [-3, 0, 3]
roots = fsolve(p, initial_guesses)
x_vals = np.linspace(-3, 3, 400)
y_vals = p(x_vals)
plt.plot(x_vals, y_vals, label='p(x) = 3x^3 - 5x^2 + 2x - 8', color='blue')
plt.axhline(0, color='black',linewidth=1)
plt.scatter(roots, p(roots), color='red', label=f'Roots: {roots}', zorder=5)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Polynomial p(x) and its Roots')
plt.legend()
plt.grid(True)
plt.show()
print(f"Roots of the polynomial: {roots}")


#ques5
# Compare the performance (time taken) of Python programs.
# 1. Convert 200MB, 400 MB, 600 MB, 800 MB, and 1000MB text files to upper case.
# 2. Generate a file with random text of all the MBs and convert it into upper case and
# check time and plot. 

import os
import time
import random
import string
import matplotlib.pyplot as plt
def generate_random_text(size_mb):
    size_bytes = size_mb * 1024 * 1024
    chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=size_bytes))
    return chars
def write_to_file(filename, text):
    with open(filename, 'w') as file:
        file.write(text)
def convert_to_uppercase(filename):
    start_time = time.time()
    with open(filename, 'r') as file:
        content = file.read()
    upper_content = content.upper()
    with open(filename, 'w') as file:
        file.write(upper_content)
    end_time = time.time()
    return end_time - start_time
sizes_mb = [200, 400, 600, 800, 1000]
time_taken = []

for size in sizes_mb:
    filename = f'random_text_{size}MB.txt'
    text = generate_random_text(size)
    write_to_file(filename, text)
    time_elapsed = convert_to_uppercase(filename)
    time_taken.append(time_elapsed)
    print(f"Time taken for {size}MB file: {time_elapsed:.2f} seconds")
plt.plot(sizes_mb, time_taken, marker='o', color='blue')
plt.xlabel('File Size (MB)')
plt.ylabel('Time Taken (seconds)')
plt.title('Time Taken to Convert Files to Uppercase')
plt.grid(True)
plt.show()


#ques6
# Consider the function f(x) = x^4−3*x^3+2.
# 1. Use SciPy to find the local minima of the function.
# 2. Plot the function f(x) over the range x = [−2,3] and mark the local minima on the plot.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
def f(x):
    return x**4 - 3*x**3 + 2
initial_guess = 0  
result = minimize(f, initial_guess)
local_minima = result.x
x_vals = np.linspace(-2, 3, 400)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label='f(x) = x^4 - 3x^3 + 2', color='blue')
plt.axhline(0, color='black',linewidth=1)
plt.scatter(local_minima, f(local_minima), color='red', label=f'Local Minima: {local_minima[0]:.2f}', zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function f(x) and Local Minima')
plt.legend()
plt.grid(True)
plt.show()
print(f"Local Minima: {local_minima}")
