from threading import Event

from message_bus import AsyncDispatcher, MessageBus


def test_async_dispatcher_invokes_handler() -> None:
    bus = MessageBus()
    bus.register("sender")
    bus.register("receiver")
    ev = Event()
    received = []

    def handler(msg):
        received.append(msg)
        ev.set()

    dispatcher = AsyncDispatcher(bus, "receiver", handler, poll_interval=0.01)
    dispatcher.start()
    bus.send("sender", "receiver", {"ping": 1})
    assert ev.wait(1.0), "handler was not invoked"
    dispatcher.stop()
    assert len(received) == 1
    assert received[0].content == {"ping": 1}
    assert received[0].sender == "sender"
    assert received[0].recipient == "receiver"
