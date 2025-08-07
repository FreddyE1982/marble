from marble_agent import MARBLEAgent
from message_bus import MessageBus
from multi_agent_env import CooperativeEnv, run_episode


def test_message_exchange():
    bus = MessageBus()
    a1 = MARBLEAgent("a1", bus=bus, eager=False)
    a2 = MARBLEAgent("a2", bus=bus, eager=False)
    a1.send("a2", {"greet": "hi"})
    msg = a2.receive(timeout=1.0)
    assert msg.content["greet"] == "hi"
    a1.broadcast({"all": 1})
    msg2 = a2.receive(timeout=1.0)
    assert msg2.content == {"all": 1}


def test_message_reply():
    bus = MessageBus()
    bus.register("a")
    bus.register("b")
    bus.send("a", "b", {"ping": 1})
    incoming = bus.receive("b", timeout=1.0)
    bus.reply(incoming, {"pong": 2})
    reply = bus.receive("a", timeout=1.0)
    assert reply.content["pong"] == 2


def test_environment_simulation_no_deadlock():
    bus = MessageBus()
    agents = {
        "a1": MARBLEAgent("a1", bus=bus, eager=False),
        "a2": MARBLEAgent("a2", bus=bus, eager=False),
    }
    env = CooperativeEnv(list(agents.keys()))
    rewards = run_episode(env, agents, steps=3)
    assert set(rewards.keys()) == {"a1", "a2"}
    # ensure episode completed without hanging and rewards are finite
    assert all(isinstance(r, float) for r in rewards.values())
