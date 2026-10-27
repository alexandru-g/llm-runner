"""``systemone-proxy`` command line: serve, spec, profiles, detect."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from systemone_proxy import __version__, native, profiles
from systemone_proxy.spec import build_spec, render_text


def _log(msg: str) -> None:
    print(f"systemone-proxy: {msg}", file=sys.stderr, flush=True)


def _resolve_profile(args):
    if args.profile and args.profile != NATIVE:
        return profiles.get(args.profile)
    if getattr(args, "gguf", None):
        model = native.NativeModel.from_gguf(args.gguf, _model_id(args.gguf))
        if model is not None:
            return model
        found = profiles.detect(args.gguf)
        if found is None:
            raise SystemExit(f"no profile matches {args.gguf}; pass --profile "
                             f"({', '.join(profiles.all_profiles())})")
        return found
    raise SystemExit("pass --profile ID or --gguf PATH")


NATIVE = "native"


def _model_id(model_path: str) -> str:
    """``vllm-d3-mini-Q8_0.gguf`` → ``vllm-d3-mini``."""
    stem = Path(model_path).stem or "native"
    return re.sub(r"[-.](?:i?q\d[\w]*|f16|bf16|f32)$", "", stem, flags=re.I).lower()


def cmd_serve(args) -> None:
    from systemone_proxy import tokenizer
    from systemone_proxy.backend import LlamaServer
    from systemone_proxy.server import serve
    from systemone_proxy.service import DecisionService

    backend = LlamaServer(args.backend, timeout=args.timeout)
    props = backend.wait_ready(args.wait_backend, log=_log)
    model_path = props.get("model_path") or ""
    if native.detect(backend):
        # llama-server then refuses the completions a profile reads, so
        # native wins over a profile that also matches (h2o's own GGUFs).
        model = native.NativeModel.from_gguf(args.gguf or model_path, _model_id(model_path))
        if model is None:
            raise SystemExit(f"llama-server serves /v1/systemone, but {model_path!r} has no "
                             "decision metadata the proxy can read; pass --gguf PATH")
        _log(f"{model.id}: served natively by llama-server (decision type "
             f"{model.decision_type}); the proxy adapts the responses")
        serve(native.NativeService(model, backend, timeout=args.timeout),
              build_spec(model, model), args.host, args.port)
        return
    if args.profile == NATIVE:
        raise SystemExit("llama-server doesn't answer /v1/systemone itself: this GGUF needs "
                         "llama.cpp b11361 or newer (vllm-d3 b11572, h2o-lightning b11773)")
    if args.profile:
        cls = profiles.get(args.profile)
    else:
        gguf = args.gguf or model_path
        cls = profiles.detect(gguf) if gguf else None
        if cls is None:
            raise SystemExit(f"no decision-model profile matches the served model {gguf!r}; "
                             f"pass --profile ({', '.join(profiles.all_profiles())})")
    n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
    if n_ctx <= 0:
        raise SystemExit("could not read the context size from llama-server /props")
    _log(f"profile {cls.id} for {model_path or args.backend} (context {n_ctx})")

    tok = tokenizer.load(cls.tokenizer_source, args.tokenizer)
    profile = cls(tok)
    if not args.skip_selftest:
        problems = profile.selftest(backend)
        if problems:
            for p in problems:
                _log(f"self-test: {p}")
            raise SystemExit("self-test failed: this GGUF does not tokenize like the vendor "
                             "model (wrong model, or a broken conversion). Use --skip-selftest "
                             "to serve anyway.")
        _log("self-test passed: GGUF tokenizer matches the vendor tokenizer")
    vision = backend.vision(props)
    if cls.image_support is not None:
        _log("images: enabled (llama-server has a vision projector)" if vision else
             "images: unavailable — llama-server runs without a vision projector (mmproj); "
             "requests with images get 503")
    service = DecisionService(profile, backend, n_ctx=n_ctx, n_probs=args.n_probs,
                              cache=args.cache, concurrency=args.concurrency, vision=vision)
    serve(service, build_spec(cls, profile), args.host, args.port)


def cmd_spec(args) -> None:
    cls = _resolve_profile(args) if (args.profile or args.gguf) else None
    spec = build_spec(cls, cls if isinstance(cls, native.NativeModel) else None)
    print(json.dumps(spec, indent=2, ensure_ascii=False) if args.json else render_text(spec))


def cmd_profiles(args) -> None:
    for pid, cls in profiles.all_profiles().items():
        print(f"{pid:<24} {cls.name:<24} {', '.join(cls.gguf_repos)}")


def cmd_detect(args) -> None:
    """A GGUF with llama.cpp decision metadata is "native"; else its profile."""
    if native.NativeModel.from_gguf(args.gguf, "x") is not None:
        print(NATIVE)
        return
    found = profiles.detect(args.gguf)
    if found is None:
        raise SystemExit(1)
    print(found.id)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="systemone-proxy", description=__doc__)
    ap.add_argument("--version", action="version", version=f"systemone-proxy {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve /v1/systemone in front of a llama-server")
    s.add_argument("--backend", required=True, help="llama-server URL, e.g. http://127.0.0.1:8080")
    s.add_argument("--profile", help="profile id, or \"native\" (default: detect from the "
                                     "served GGUF; native when llama-server answers "
                                     "/v1/systemone itself)")
    s.add_argument("--gguf", help="GGUF path to detect the profile from (default: llama-server's)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--tokenizer", help="local tokenizer.json instead of the pinned download")
    s.add_argument("--n-probs", type=int, default=128,
                   help="top-n logprobs requested per call (raised to labels + 64)")
    s.add_argument("--cache", choices=["off", "request"], default="off",
                   help="llama-server prompt-cache reuse: off (default, every call computed "
                        "fresh) or within one request only")
    s.add_argument("--concurrency", type=int, default=1,
                   help="backend calls in flight per request (match llama-server --parallel)")
    s.add_argument("--timeout", type=float, default=600.0, help="per backend call, seconds")
    s.add_argument("--wait-backend", type=float, default=900.0,
                   help="seconds to wait for llama-server to load at startup")
    s.add_argument("--skip-selftest", action="store_true")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("spec", help="print the client contract (optionally for one model)")
    s.add_argument("--profile")
    s.add_argument("--gguf")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_spec)

    s = sub.add_parser("profiles", help="list supported models")
    s.set_defaults(func=cmd_profiles)

    s = sub.add_parser("detect", help="print the profile id for a GGUF (exit 1 if none)")
    s.add_argument("gguf")
    s.set_defaults(func=cmd_detect)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
