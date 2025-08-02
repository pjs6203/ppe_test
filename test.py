five = 0
three = 0



for i in range(1, 1000):
    if (i % 3) == 0:
        three = three + i

    if (i % 5) == 0:
        five = five + i

print(three)
print(five)
print(five + three)