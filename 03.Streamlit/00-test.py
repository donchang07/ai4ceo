import streamlit as st

st.title("Python 예제 결과")

# 1부터 100까지의 숫자 합계
st.header("1. 1부터 100까지의 숫자 합계")
total = sum(range(1, 101))
st.write(f"합계: {total}")

# 짝수 구하기 함수
def get_even_numbers(n):
    return [i for i in range(1, n+1) if i % 2 == 0]

# 1부터 10까지의 짝수 출력
st.header("2. 1부터 10까지의 짝수")
even_numbers = get_even_numbers(10)
st.write(even_numbers)

# Person 클래스 정의
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"이름: {self.name}, 나이: {self.age}"

# Person 객체 리스트 생성 및 출력
st.header("3. Person 객체 리스트")
plist = [Person("영희", 25), Person("철수", 31), Person("영식", 22)]
for person in plist:
    st.write(str(person))

# 로마 숫자를 정수로 변환하는 함수
def romanToInt(s: str) -> int:
    roman = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    result = 0
    for i in range(len(s)):
        if i > 0 and roman[s[i]] > roman[s[i - 1]]:
            result += roman[s[i]] - 2 * roman[s[i - 1]]
        else:
            result += roman[s[i]]
    return result

# 로마 숫자 변환 결과 출력
st.header("4. 로마 숫자를 정수로 변환")
roman_examples = ["III", "IV", "IX"]
for roman in roman_examples:
    st.write(f"{roman} -> {romanToInt(roman)}")

# 정수를 로마 숫자로 변환하는 함수
def intToRoman(num: int) -> str:
    roman = {
        1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X',
        40: 'XL', 50: 'L', 90: 'XC', 100: 'C',
        400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'
    }
    result = ""
    for value, letter in sorted(roman.items(), key=lambda x: x[0], reverse=True):
        while num >= value:
            result += letter
            num -= value
    return result

# 정수를 로마 숫자로 변환한 결과 출력
st.header("5. 정수를 로마 숫자로 변환")
int_examples = [3, 400]
for num in int_examples:
    st.write(f"{num} -> {intToRoman(num)}")