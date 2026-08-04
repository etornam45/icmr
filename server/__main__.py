"""python -m server"""

from server.config import load_config


def main() -> None:
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "server.app:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
