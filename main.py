"""
Entry point. Run with:
  python main.py            # dashboard only (manual scans via UI)
  python main.py --agent    # dashboard + background agent
  python main.py --scan     # one-shot scan then exit
"""
import asyncio
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Resellify — Market resell agent")
    parser.add_argument("--agent", action="store_true", help="Start background scanning agent")
    parser.add_argument("--scan", action="store_true", help="Run a single scan and exit")
    parser.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", 8000)))
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    if args.scan:
        from src.agent import load_config, run_scan, init_db
        from src.db.models import init_db as db_init

        async def _scan():
            await db_init()
            config = load_config(args.config)
            good = await run_scan(config)
            print(f"Scan complete. Good deals: {good}")

        asyncio.run(_scan())
        return

    import uvicorn
    from src.dashboard.app import app, _agent_task
    from src.agent import start_agent

    if args.agent:
        async def _run_with_agent():
            from src.db.models import init_db as db_init
            await db_init()
            agent_task = asyncio.create_task(start_agent(args.config))
            config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
            agent_task.cancel()

        asyncio.run(_run_with_agent())
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
