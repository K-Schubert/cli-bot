from __future__ import annotations

import logging
from typing import Optional

from .config import load_service_settings, parse_args
from .services import build_clients, build_tooling, prepare_store
from .session import run_entrypoint


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    noisy_packages = ("httpx", "httpcore", "urllib3", "chromadb")
    for name in noisy_packages:
        logging.getLogger(name).setLevel(logging.WARNING)

    if verbose:
        logging.getLogger("cli").setLevel(logging.DEBUG)
        logging.getLogger("ToolRunner").setLevel(logging.DEBUG)
        logging.getLogger("RAGTool").setLevel(logging.DEBUG)
        logging.getLogger("PDFUpsert").setLevel(logging.DEBUG)
        logging.getLogger("ConversationHistory").setLevel(logging.DEBUG)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(bool(args.verbose))

    settings = load_service_settings(args)
    clients = build_clients(settings, args)
    store = prepare_store(args.collection, args.csv, verbose=bool(args.verbose))
    tooling = build_tooling(
        store=store,
        clients=clients,
        doc_validation=args.doc_validation,
        search_k=args.search_k,
        verbose=bool(args.verbose),
    )

    run_entrypoint(args, clients, store, tooling)


if __name__ == "__main__":
    main()
