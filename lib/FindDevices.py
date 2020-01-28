'''
Automatically chooses a device for the user
'''
from tensorflow.python.client import device_lib

def getAvaliableDevicesToTrain(deviceType='CPU'):
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos if x.device_type == deviceType]
    return devices

def getChosenDevice(deviceType='GPU', deviceNum=0):
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos if x.device_type == deviceType]
    numberOfDevicesFound = len(devices)
    try:
        if len(devices) == 0:
            return getAvaliableDevicesToTrain(deviceType='CPU')[deviceNum]
        else:
            return devices[deviceNum]
    except:
        print("Device number is in valid")

