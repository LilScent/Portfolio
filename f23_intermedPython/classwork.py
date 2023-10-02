"""
Classwork

Author: Drew Xavier
"""

import copy

"""If an object is immutable and identical to another object both labels will point to the same object stored in 
memory"""
object_1 = "immutable string"
object_2 = "immutable string"
print(id(object_1))
print(id(object_2))
print(object_1 is object_2)

cloud_drive = ["word", "powerpoint", "excel"]
lynx_computer = ["photoshop", cloud_drive]
eric_computer = ["minecraft", cloud_drive]

lynx_other_computer = copy.copy(lynx_computer)

print(lynx_computer)
print(eric_computer)
print(lynx_other_computer)

cloud_drive.append("Resume")
lynx_other_computer[1].remove("powerpoint")

print(lynx_computer)
print(eric_computer)
print(lynx_other_computer)