"""
zip 是 Python 中的一个内置函数，它接受任意数量的可迭代对象（例如列表、元组等）作为输入，并将这些对象中对应的元素打包成一个个元组，
然后返回由这些元组组成的对象。如果各个可迭代对象的元素个数不一致，则 zip 函数返回的对象长度与最短的对象相同。

下面是一些使用 zip 函数的例子：
"""
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
zipped = zip(list1, list2)

# 输出：[(1, 'a'), (2, 'b'), (3, 'c')]
print(list(zipped))


# 在这种情况下，由于 list1 比 list2 长，所以 zip 只打包了 list2 中存在的元素对应的 list1 中的元素。
list1 = [1, 2, 3, 4]
list2 = ['a', 'b', 'c']
zipped = zip(list1, list2)

# 输出：[(1, 'a'), (2, 'b'), (3, 'c')]
print(list(zipped))



# zip 常与 for 循环一起使用，以并行遍历多个列表：
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

for x, y in zip(list1, list2):
    print(x, y)

# 输出：
# 1 a
# 2 b
# 3 c

# 你还可以使用 * 运算符来解包 zip 生成的元组，并分别赋值给多个变量：
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
x, y = zip(*zip(list1, list2))

# 输出：((1, 2, 3), ('a', 'b', 'c'))
print(x, y)

"""
在这个例子中，zip(*zip(list1, list2)) 首先使用内部的 zip 将 list1 和 list2 打包成元组，然后使用外部的 zip
和 * 运算符将打包后的元组解包，并以相反的顺序重新组合成两个新的元组。
总之，zip 函数是一个非常实用的工具，它可以让你并行地处理多个可迭代对象中的元素。
"""