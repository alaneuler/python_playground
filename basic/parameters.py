def f(*, arg1, arg2):
    print(arg1, arg2)


# This one won't work.
# f(1, 2)
f(arg1=1, arg2=2)
