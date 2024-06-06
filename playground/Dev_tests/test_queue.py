from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
import time

class MyEventHandler(PatternMatchingEventHandler):
    def on_created(self,event):
        print(f"hey, {event.src_path} has been created!")
    def on_deleted(self,event):
        print(f"what the f**k! Someone deleted {event.src_path}!")
# Create an event handler
event_handler = MyEventHandler()

# Replace with your directory path
watch_dir = "./../Transform_Queue/"

observer = Observer()
# The recursive=False makes so we monitor only the directory, not subdirectories
observer.schedule(event_handler, watch_dir, recursive=False)  
observer.start()

try:
  # Keep the script running to monitor changes
  while True:
        time.sleep(1)
except KeyboardInterrupt:
  observer.stop()
  observer.join()