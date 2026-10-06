import argparse
import asyncio
import logging
import os
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(prog="soul", description="One model, many machines")
    commands = parser.add_subparsers(dest="command", required=True)
    controller = commands.add_parser("controller", help="Run the controller and admin UI")
    controller.add_argument("--host", default="127.0.0.1")
    controller.add_argument("--port", type=positive, default=8000)
    controller.add_argument("--state-dir", type=Path, default=Path(".soul/controller"))
    controller.add_argument(
        "--dev", action="store_true", help="Loopback-only development with soul-dev credentials"
    )
    controller.add_argument("--max-context", type=positive, default=2048)
    controller.add_argument("--max-queue", type=positive, default=32)
    controller.add_argument("--queue-timeout", type=positive, default=300)
    controller.add_argument("--rpc-timeout", type=positive, default=120)
    controller.add_argument("--load-timeout", type=positive, default=1800)
    controller.add_argument("--max-layers-per-node", type=positive)
    controller.add_argument("--allow-local-models", action="store_true")
    controller.add_argument("--allow-model", action="append", default=[])
    node = commands.add_parser("node", help="Join a controller as an inference node")
    node.add_argument("--controller", default="http://127.0.0.1:8000")
    node.add_argument("--node-id")
    node.add_argument("--name")
    node.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/soul")
    node.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    node.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    node.add_argument("--memory-gb", type=float, help="Upper bound on the node memory budget")
    node.add_argument("--weight-mode", choices=("resident", "stream"), default="resident")
    node.add_argument(
        "--dev", action="store_true", help="Use the local controller's soul-dev node credential"
    )
    demo = commands.add_parser(
        "demo", help="Start a controller and two CPU nodes with a tiny untrained model"
    )
    demo.add_argument("--port", type=positive, default=8000)
    demo.add_argument("--state-dir", type=Path, default=Path(".soul/demo"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "controller":
        import uvicorn

        from soul.controller.app import Settings, create_app

        if args.dev and args.host not in ("127.0.0.1", "localhost", "::1"):
            parser.error("--dev must bind to a loopback address")
        keys = {
            name: "soul-dev" if args.dev else os.environ.get(env, "")
            for name, env in (
                ("api_key", "SOUL_API_KEY"),
                ("admin_key", "SOUL_ADMIN_KEY"),
                ("node_token", "SOUL_NODE_TOKEN"),
            )
        }
        if not all(keys.values()):
            parser.error("Set SOUL_API_KEY, SOUL_ADMIN_KEY and SOUL_NODE_TOKEN (or use --dev locally)")
        settings = Settings(
            **keys,
            state_dir=args.state_dir,
            max_context=args.max_context,
            max_queue=args.max_queue,
            queue_timeout=args.queue_timeout,
            rpc_timeout=args.rpc_timeout,
            load_timeout=args.load_timeout,
            allow_local=args.allow_local_models,
            max_layers_per_node=args.max_layers_per_node,
            allowed_models=args.allow_model,
        )
        logging.info("Admin UI: http://%s:%s", args.host, args.port)
        uvicorn.run(
            create_app(settings),
            host=args.host,
            port=args.port,
            ws_max_size=64 * 1024**2,
            ws_per_message_deflate=False,
            timeout_graceful_shutdown=15,
        )
    elif args.command == "demo":
        from soul.demo import run

        run(args.port, args.state_dir)
    else:
        from urllib.parse import urlparse

        if args.dev and urlparse(args.controller).hostname not in ("127.0.0.1", "localhost", "::1"):
            parser.error("--dev only connects to a loopback controller")
        token = "soul-dev" if args.dev else os.environ.get("SOUL_NODE_TOKEN")
        if not token:
            parser.error("Set SOUL_NODE_TOKEN (or use --dev locally)")
        if args.memory_gb is not None and args.memory_gb <= 0:
            parser.error("--memory-gb must be positive")
        try:
            from soul.node.client import NodeClient
        except ImportError:
            parser.error(
                "Install node dependencies: pip install 'soul-serving[node]' (or pip install '.[node]')"
            )
        client = NodeClient(
            args.controller,
            token,
            args.cache_dir,
            node_id=args.node_id,
            name=args.name,
            device=args.device,
            memory_gb=args.memory_gb,
            weight_mode=args.weight_mode,
            dtype=args.dtype,
        )
        try:
            asyncio.run(client.run())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
