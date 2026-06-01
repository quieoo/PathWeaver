import json
import os
import threading
import time
from typing import Any, Dict, Optional


if os.getenv("PATHWEAVER_ENABLE_LMCACHE_BREAKDOWN") == "1":
    _BREAKDOWN_FILE = os.getenv("PATHWEAVER_LMCACHE_BREAKDOWN_FILE")
    _LOCK = threading.Lock()
    import torch

    def _append_record(record: Dict[str, Any]) -> None:
        if not _BREAKDOWN_FILE:
            return
        payload = json.dumps(record, ensure_ascii=True, sort_keys=True)
        with _LOCK:
            with open(_BREAKDOWN_FILE, "a", encoding="utf-8") as fh:
                fh.write(payload)
                fh.write("\n")

    try:
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
        from lmcache.v1.cache_engine import LMCacheEngine
        from lmcache.v1.compute.blend.blender import LMCBlender
    except Exception:
        LMCacheConnectorV1Impl = None
        LMCacheEngine = None
        LMCBlender = None

    if LMCacheConnectorV1Impl is not None:
        _orig_get_num_new_matched_tokens = LMCacheConnectorV1Impl.get_num_new_matched_tokens
        _orig_start_load_kv = LMCacheConnectorV1Impl.start_load_kv

        def _patched_get_num_new_matched_tokens(self, request, num_computed_tokens):
            t0 = time.perf_counter()
            result = _orig_get_num_new_matched_tokens(self, request, num_computed_tokens)
            elapsed = time.perf_counter() - t0

            load_spec = self.load_specs.get(request.request_id)
            lmcache_cached_tokens = None
            vllm_cached_tokens = num_computed_tokens
            if load_spec is not None:
                lmcache_cached_tokens = int(load_spec.lmcache_cached_tokens)
                vllm_cached_tokens = int(load_spec.vllm_cached_tokens)

            _append_record(
                {
                    "event": "scheduler_lookup",
                    "external_tokens_to_load": None if result is None else int(result),
                    "lmcache_cached_tokens": lmcache_cached_tokens,
                    "lookup_time_s": elapsed,
                    "prompt_tokens": int(getattr(request, "num_tokens", 0)),
                    "req_id": request.request_id,
                    "ts": time.time(),
                    "vllm_cached_tokens": vllm_cached_tokens,
                }
            )
            return result

        def _patched_start_load_kv(self, forward_context, **kwargs):
            self.current_layer = 0

            if len(self.kv_caches) == 0:
                self._init_kv_caches_from_forward_context(forward_context)

            metadata = self._parent._get_connector_metadata()
            assert len(self.kv_caches) > 0
            kvcaches = list(self.kv_caches.values())

            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return

            assert self.lmcache_engine is not None

            self.lmcache_engine.post_init(kvcaches=kvcaches)

            self.layerwise_retrievers = []

            last_idx = None
            for idx, request in enumerate(metadata.requests):
                if request.load_spec is not None:
                    last_idx = idx

            for idx, request in enumerate(metadata.requests):
                if request.load_spec is None:
                    continue

                req_id = getattr(request, "req_id", None)
                tokens = request.token_ids
                slot_mapping = request.slot_mapping.cuda()
                assert len(tokens) == len(slot_mapping)

                token_mask = torch.ones(len(tokens), dtype=torch.bool)
                masked_token_count = (
                    request.load_spec.vllm_cached_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                token_mask[:masked_token_count] = False

                lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
                if self.use_layerwise:
                    sync = idx == last_idx
                    if self.enable_blending:
                        self.blender.blend(
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            kvcaches=kvcaches,
                            slot_mapping=slot_mapping[:lmcache_cached_tokens],
                            req_id=req_id,
                        )
                    else:
                        layerwise_retriever = self.lmcache_engine.retrieve_layer(
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            kvcaches=kvcaches,
                            slot_mapping=slot_mapping[:lmcache_cached_tokens],
                            sync=sync,
                            req_id=req_id,
                        )
                        next(layerwise_retriever)
                        next(layerwise_retriever)
                        self.layerwise_retrievers.append(layerwise_retriever)
                else:
                    ret_token_mask = self.lmcache_engine.retrieve(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        request_configs=request.request_configs,
                        req_id=req_id,
                        skip_contains_check=True,
                    )

                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(len(tokens))

        LMCacheConnectorV1Impl.get_num_new_matched_tokens = _patched_get_num_new_matched_tokens
        LMCacheConnectorV1Impl.start_load_kv = _patched_start_load_kv

    if LMCacheEngine is not None:
        _orig_retrieve = LMCacheEngine.retrieve
        _orig_retrieve_layer = LMCacheEngine.retrieve_layer

        def _patched_retrieve(self, tokens, mask=None, **kwargs):
            req_id = kwargs.get("req_id")
            t0 = time.perf_counter()
            ret = _orig_retrieve(self, tokens, mask=mask, **kwargs)
            elapsed = time.perf_counter() - t0
            if req_id is not None:
                retrieved_tokens = None
                try:
                    retrieved_tokens = int(ret.sum().item())
                except Exception:
                    retrieved_tokens = None
                _append_record(
                    {
                        "event": "worker_retrieve",
                        "kv_load_time_s": elapsed,
                        "req_id": req_id,
                        "retrieved_tokens": retrieved_tokens,
                        "ts": time.time(),
                    }
                )
            return ret

        def _patched_retrieve_layer(self, tokens, mask=None, **kwargs):
            req_id = kwargs.get("req_id")
            t0 = time.perf_counter()
            gen = _orig_retrieve_layer(self, tokens, mask=mask, **kwargs)
            expected_yields = int(getattr(self, "num_layers", 0)) + 2

            def _wrapped():
                yield_count = 0
                try:
                    while True:
                        item = next(gen)
                        yield_count += 1
                        if req_id is not None and yield_count == expected_yields:
                            elapsed = time.perf_counter() - t0
                            retrieved_tokens = None
                            try:
                                if item is not None:
                                    retrieved_tokens = int(item.sum().item())
                            except Exception:
                                retrieved_tokens = None
                            _append_record(
                                {
                                    "event": "worker_retrieve_layerwise",
                                    "kv_load_time_s": elapsed,
                                    "req_id": req_id,
                                    "retrieved_tokens": retrieved_tokens,
                                    "ts": time.time(),
                                }
                            )
                        yield item
                except StopIteration:
                    return

            return _wrapped()

        LMCacheEngine.retrieve = _patched_retrieve
        LMCacheEngine.retrieve_layer = _patched_retrieve_layer

    if LMCBlender is not None:
        _orig_blend = LMCBlender.blend

        def _patched_blend(self, tokens, mask=None, **kwargs):
            req_id = kwargs.get("req_id")
            t0 = time.perf_counter()
            ret = _orig_blend(self, tokens, mask=mask, **kwargs)
            elapsed = time.perf_counter() - t0
            if req_id is not None:
                _append_record(
                    {
                        "blend_total_time_s": elapsed,
                        "event": "worker_blend_total",
                        "req_id": req_id,
                        "ts": time.time(),
                    }
                )
            return ret

        LMCBlender.blend = _patched_blend
