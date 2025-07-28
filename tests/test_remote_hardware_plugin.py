import grpc
from concurrent import futures
from remote_hardware import load_remote_tier_plugin

class EchoHandler(grpc.GenericRpcHandler):
    def service(self, handler_call_details):
        def unary(request, context):
            return request
        return grpc.unary_unary_rpc_method_handler(unary)

def test_grpc_plugin_offload():
    server = grpc.server(futures.ThreadPoolExecutor())
    handler = EchoHandler()
    server.add_generic_rpc_handlers((handler,))
    port = server.add_insecure_port("localhost:0")
    server.start()
    addr = f"localhost:{port}"
    tier = load_remote_tier_plugin(
        "remote_hardware.grpc_tier", address=addr
    )
    tier.connect()
    out = tier.offload_core(b"abc")
    assert out == b"abc"
    tier.close()
    server.stop(0)
