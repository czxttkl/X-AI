"""
Test multiprocessing object sharing
"""
from multiprocessing import Process, Manager, Array
from multiprocessing.managers import BaseManager
import numpy
import tensorflow as tf

print('process execute to here')

class GetSetter(object):
    def __init__(self):
        self.var = None

    def set(self, value):
        self.var = value

    def get(self):
        return self.var


class ChildClass(object):
    def __init__(self):
        self.v = numpy.array([44])


class ParentClass(GetSetter):
    def __init__(self, v):
        print('init parentclass')
        self.sess = tf.Session()
        self.w = tf.get_variable('w', [5], initializer=tf.constant_initializer(2.))
        self.s = tf.placeholder(tf.float32, [5], name='s')
        self.output = self.w + self.s
        self.sess.run(tf.global_variables_initializer())

        self.v = v
        self.child = ChildClass()
        GetSetter.__init__(self)
        self.array = [[numpy.ones(5), True, numpy.zeros(5)], [numpy.zeros(5), False, numpy.ones(5)]]

    def getChild(self):
        return self.child

    def getValue(self):
        return self.v

    def getArray(self):
        return self.array

    def changeArray(self):
        self.array[0][0][0] = 2
        return self.array

    def getArrayShape(self):
        return self.array[0][0].shape

    @property
    def vv(self):
        return self.v

    def getV(self):
        return self.v

    def getW(self, ss):
        return self.sess.run(self.output, feed_dict={self.s: ss})

    def getSess(self):
        return self.sess

def change_obj_value(obj, ss):
    print('process', obj.getW(ss))


if __name__ == '__main__':
    BaseManager.register('ParentClass', ParentClass)
    manager = BaseManager()
    manager.start()
    inst2 = manager.ParentClass(v=7)

    ss = numpy.array([1,2,3,4,5])
    p2 = Process(target=change_obj_value, args=[inst2, ss])
    p2.start()
    p2.join()

    print('self memory location:', inst2)                    # <__main__.ParentClass object at 0x10cf82350>
    print('self.Child memory location:', inst2.getChild())         # <__main__.ChildClass object at 0x10cf6dc50>
    # print('self.v:', inst2.v)                 # fail
    # print('self.vv:', inst2.vv)             # fail
    # print('self.getSess():', inst2.getSess())  # fail
    print('self.getV():', inst2.getV())
    print('self.getValue():', inst2.getValue())
    print('self.getArray():', inst2.getArray())
    print('self.changeArray():', inst2.changeArray())
    print('self.getArrayShape():', inst2.getArrayShape())
    print('self.getChild().v:', inst2.getChild().v)
    #good!
    arr = Array('i', range(10))
    print(numpy.max(arr))

