from websockets.sync.client import connect

with connect("ws://localhost:8000") as websocket:
    websocket.send("Eureka!")
    assert websocket.recv() == "Eureka!"

    websocket.send("I've found it!")
    assert websocket.recv() == "I've found it!"
