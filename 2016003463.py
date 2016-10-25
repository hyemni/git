"""
===========================
  Rock Scissors Paper Game
===========================

1. Created by : Nam Hyemin
2. Date : 2016.10.12

========== ========== =======
 computer     user    result
========== ========== =======
Rock          Rock	Draw
Rock        Scissors	Lose
Rock	      Paper	Win
Scissors      Rock	Win
Scissors    Scissors	Draw
Scissors      Paper	Lose
Paper	      Rock	Lose
Paper       Scissors	Win
Paper	      Paper	Draw
========== ========== =======
"""

import random

def main():	
	for i in range(10):
		computer = random.randrange(3) + 1
		user = int(input("1.가위, 2.바위, 3.보를 입력하세요. "))
		while not(user == 1 or user == 2 or user ==3):
			user = int(input("1.가위, 2.바위, 3.보를 입력하세요. "))
		
		if(computer == 1):
			if(user == 1):
				print("컴퓨터: 가위 -----> 비김!")
			elif(user == 2):
				print("컴퓨터: 가위 -----> 사용자 승~")
			elif(user == 3):
				print("컴퓨터: 가위 -----> 컴퓨터 승ㅜ")
		
		elif(computer == 2):
			if(user == 1):
				print("컴퓨터: 바위 -----> 컴퓨터 승ㅜ")
			elif(user == 2):
				print("컴퓨터: 바위 -----> 비김!")
			elif(user == 3):
				print("컴퓨터: 바위 -----> 사용자 승~")
		
		elif(computer == 3):
			if(user == 1):
				print("컴퓨터: 보 -----> 사용자 승~")
			elif(user == 2):
				print("컴퓨터: 보 -----> 컴퓨터 승ㅜ")
			elif(user == 3):
				print("컴퓨터: 보 -----> 비김!")

main()
