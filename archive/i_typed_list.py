import collections


class ITypedList(collections.MutableSequence):

    def __init__(self, allowed_types, *args):
        self.type = allowed_types
        self.list = list()
        self.extend(list(args))

    def check(self, v):
        if not isinstance(v, self.type):
            raise TypeError(f'Does not support {type(v)}. Use {self.type} instead.')

    def __len__(self): return len(self.list)

    def __getitem__(self, i): return self.list[i]

    def __delitem__(self, i): del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def append(self, v):
        self.check(v)
        self.list.append(v)

    def __str__(self):
        return self.list.__str__()
