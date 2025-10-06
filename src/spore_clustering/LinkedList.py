#Nodes maintain their next and prev pointers after being popped for
# the convenience of popping while iterating and still having access to the next/prev
from __future__ import annotations
import typing as tp

T = tp.TypeVar('T')
T2 = tp.TypeVar('T2')
class LinkedList(tp.Generic[T]):
    class Node(tp.Generic[T2]):
        def __init__(self, data: T2 = None, next: LinkedList.Node[T2] = None, prev: LinkedList.Node[T2] = None):
            self.data = data
            self.next = next
            self.prev = prev
        
        def get(self): return self.data

        def delete(self): self.data = None
    
    def __init__(self, iterable: tp.Iterable[T] = []):
        self.head: LinkedList.Node[T] = None
        self.tail: LinkedList.Node[T] = None
        self.len = 0
        self.extend(iterable)

    def __len__(self): return self.len

    def __bool__(self): return (self.len > 0)

    def get(self, index: int):
        if not self.len:
            return None
        if index < 0:
            curr = self.tail
            for _ in range(abs(index) - 1): 
                if not curr:
                    return None
                curr = curr.prev
            return curr
        curr = self.head
        for _ in range(index): 
            if not curr:
                return None
            curr = curr.next
        return curr

    def append(self, data: T):
        if not self.len:
            self.head = self.Node(data, None, None)
            self.tail = self.head
            self.len = 1
            return
        self.tail.next = self.Node(data, None, self.tail)
        self.tail = self.tail.next
        self.len += 1

    def extend(self, iterable: tp.Iterable[T]):
        for item in iterable:
            self.append(item)

    def insert(self, index: tp.Union[LinkedList.Node[T], int, None], data: T):
        if index == None or index == self.len:
            self.append(data)
            return
        if not self.len:
            if index == 0 or index == -1:
                self.head = self.Node(data, None, None)
                self.tail = self.head
                self.len = 1
            else: 
                raise ValueError("Insertion index out of range")
            return
        index: LinkedList.Node[T] = index if isinstance(index, self.Node) else self.get(index)
        node = self.Node(data, index, index.prev)
        if index is self.head:
            self.head = node
        else:
            index.prev.next = node
        index.prev = node
        self.len += 1

    def pop(self, index: tp.Union[LinkedList.Node[T], int], secure_pop: bool = True):
        if not self.len:
            return
        index: LinkedList.Node[T] = index if isinstance(index, self.Node) else self.get(index)
        data = index.data
        if index is self.head:
            self.head = self.head.next
            if not self.head:
                self.tail = None
            else:
                self.head.prev = None
        elif index is self.tail:
            self.tail = self.tail.prev
            self.tail.next = None
        else:
            index.prev.next = index.next
            index.next.prev = index.prev
        if secure_pop:
            index.prev, index.next = None, None
        self.len -= 1
        return data

    def clear(self):
        self.head = self.tail = None
        self.len = 0
    
    def copy(self):
        copy = LinkedList()
        for curr in self:
            copy.append(curr.data)
        return copy
    
    def __iter__(self):
        curr = self.head
        while curr:
            yield curr
            curr = curr.next
    
    def __repr__(self):
        s = '<LinkedList(['
        n = 0
        curr = self.head
        while n < 1000 and curr:
            s += repr(curr.data) + ', '
            curr = curr.next
            n += 1
        if n >= 1000 and curr:
            s += '...'
        return s + '])>'

    def __str__(self):
        return self.__repr__()
