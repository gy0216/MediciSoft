import time
import sys

import stomp

conn = stomp.Connection()

conn.start()
conn.connect('admin', 'password', wait=True)

conn.send('/queue/testQueue', "Send Python message to queue")
print("Send Python message to queue")

time.sleep(2)
conn.disconnect()
