# print("question 1");
# number = int(input("Your number: "))

# if(number % 2 == 0):
#     print("Your number is even")
# else:
#     print("Your number is odd")


# print("question 2")
# numList = []

# for i in range(3):
#     number = int(input("Your number: "))
#     numList.append(number)

# if(numList[0] > numList[1] and numList[0] > numList[2]):
#     print(numList[0])
# elif(numList[1] > numList[0] and numList[1] > numList[2]):
#     print(numList[1])
# else:
#     print(numList[2])


# print("question 3");
# limit = int(input("Your limit: "))

# if(limit == 0):
#     print(1);
# else:
#     fact = 1;
#     for i in range(limit):
#         fact = fact * (i+1);

# print(fact);    



# print("question 4");
# str = input("Your string: ")
# print(str[::-1])


# print("question 5");
# str = input("Your string: ")
# if(str == str[::-1]):
#     print("Your string is a palindrome")
# else:
#     print("Your string is not a palindrome")



# print("question 6");
# fibonacci = [0, 1]
# limit = int(input("Your limit: "))
# for i in range(2, limit):
#     fibonacci.append(fibonacci[i-1] + fibonacci[i-2])
# print(fibonacci)


# print("question 7");
# number = int(input("Your number: "))

# for i in range(2, number):
#     if(number % i == 0):
#         print("Your number is not a prime number")
#         break
# else:
#     print("Your number is a prime number")

     
# print("question 8");
# limit = int(input("Your limit: "))
# numList = []

# for i in range(limit):
#     number = int(input("Your number: "))
#     numList.append(number)

# for i in numList:
#     if(i > 500):
#         break
#     elif(i > 150):
#         continue
#     elif(i % 5 == 0):
#         print(i)



print("question 9");
number = int(input("Your number: "))

count = 0; 

while(number > 0):
    count += 1;
    number = number // 10;

print(count);


