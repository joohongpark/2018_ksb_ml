#coding=utf-8
import serial
import json
import threading
import RPi.GPIO as GPIO
import urllib2

GPIO.setmode(GPIO.BCM)
GPIO.setup(4 ,GPIO.IN)

url = "http://192.168.0.16:53001"  # URL

DEV_PATH = '/dev/ttyUSB0'
#DEV_PATH = '/dev/tty.usbserial-DA00HZQQ'


def READ_Serial(ser):
    global count
    while True:
        try:
            jsondata = json.loads(comm.readline())
            for k in jsondata:
                if (jsondata[k][u'mode'] == u'GET_POW') and (int(jsondata[k][u'power_count']) >= 1) :
                    jsondata[k][u'error'] = GPIO.input(4)
                    #fp = open("./d.txt", 'r')
                    params = json.dumps(jsondata[k]).encode("utf-8")

                    req = urllib2.Request(url)
                    req.add_header('Content-Type', 'application/json')
                    response = urllib2.urlopen(req, params)
                    print(response.read().decode('utf8'))
                    #print json.dumps(jsondata[k])
                    #fp.close()
        except Exception as e:
            print(e)
    conn.close()

try:
    comm = serial.Serial(DEV_PATH, baudrate = 115200)
    comm.flushInput()
except:
    print("시리얼 접속 에러")
    exit()
comm.writelines('$S0901' + chr(0x0D))
comm.writelines('$S0901' + chr(0x0D))
comm.writelines('$C091' + chr(0x0D))
thread1 = threading.Thread(target=READ_Serial, args=(comm,))
thread1.start()
