# 2-1
a = "你好"
print(a)
# 2-2
a = "西红柿"
print(a)
a = "土豆"
print(a)
# 2-3
name = "张三"
print('Hello {}, would you like to learn some Python today?'.format(name))
print('Hello %s, would you like to learn some Python today?' % name)
# 2-4
name = "alice"
print(name.upper())  # 转小写
print(name.lower())  # 转大写
print(name.capitalize())  # 字符串首字母大写
print(name.title())  # 字符串中所有单词首字母大学
# 2-5
print('Albert Einstein once said, “A person who never made a mistake never tried anything new.”')
# 2-6
famous_person = "Albert Einstein"
message = "A person who never made a mistake never tried anything new."
print('{0} once said, “{1}”'.format(famous_person, message))
# 2-7
name = "   \t张三   \n"
print(name)
print(name.lstrip())  # 清除左边空格
print(name.rstrip())  # 清除右边空格
print(name.strip())  # 清除两边空格
# 2-9
a = 6
print("我最喜欢的数字是{}".format(a))
# 2-10
name = "   \t张三   \n"
print(name)
print(name.lstrip())  # 清除左边空格
print(name.rstrip())  # 清除右边空格
print(name.strip())  # 清除两边空格
'使用lstrip()、rstrip()、strip()方法清楚字符串的空格'
name = "alice"
print(name.upper())  # 转小写
print(name.lower())  # 转大写
print(name.capitalize())  # 字符串首字母大写
print(name.title())  # 字符串中所有单词首字母大学
"对字符串的大小写进行转换"
# 3-1
names = ["张三", "李四", "王五"]
for i in names:
    print(i)
# 3-2
names = ["张三", "李四", "王五"]
for i in names:
    print(i, "吃饭了吗？", sep="")
# 3-3
list1 = ["单车", "汽车", "地铁"]
print("{}出行绿色健康".format(list1[0]))
print("{}出行方便快捷".format(list1[1]))
print("{}出行避免堵车".format(list1[2]))
# 3-4
names = ["乔布斯", "马云", "巴菲特"]
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
# 3-5
names = ["乔布斯", "马云", "巴菲特"]
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
print("乔布斯无法赴约")
names[0] = "比尔盖茨"
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
# 3-6
names = ["乔布斯", "马云", "巴菲特"]
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
print("乔布斯无法赴约")
names[0] = "比尔盖茨"
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
print("我找到了一个更大的餐桌")
names.insert(0,"马化腾")
names.insert(2,"王建林")
names.append("李彦宏")
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
#3-7
names = ["乔布斯", "马云", "巴菲特"]
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
print("乔布斯无法赴约")
names[0] = "比尔盖茨"
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
print("我找到了一个更大的餐桌")
names.insert(0,"马化腾")
names.insert(2,"王建林")
names.append("李彦宏")
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
print("抱歉，刚买的餐桌无法及时送达")
print("{}抱歉，我们无法共进晚餐".format(names[0]))
names.pop(0)
print("{}抱歉，我们无法共进晚餐".format(names[1]))
names.pop(1)
print("{}抱歉，我们无法共进晚餐".format(names[2]))
names.pop(2)
print("{}抱歉，我们无法共进晚餐".format(names[2]))
names.pop(2)
for i in names:
    print(i, "依然可以来共进晚餐", sep="")
del names[0:2]
print(names)
#3-8
list1=["france","america","english","denmark","iceland"]
print(list1)
print(sorted(list1))
print(list1)
print(sorted(list1,reverse = True))
print(list1)
list1.reverse()
print(list1)
list1.reverse()
print(list1)
list1.sort()
print(list1)
list1.sort(reverse=True)
print(list1)
#3-9
names = ["乔布斯", "马云", "巴菲特"]
for i in names:
    print(i, "可以与我共进晚餐吗", sep="")
print("我共邀请了{}名嘉宾".format(len(names)))